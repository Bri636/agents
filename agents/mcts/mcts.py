""" Credit to: https://github.com/maitrix-org/llm-reasoners """
from __future__ import annotations

import pickle
from os import PathLike
import pickle
import math
from copy import deepcopy
from typing import Generic, Optional, NamedTuple, Callable, Hashable
import itertools
from abc import ABC
from collections import defaultdict
from rich.console import Console
from rich.table import Table
from io import StringIO
import numpy as np
from tqdm import trange

from agents.algorithms.tree_search.base import (SearchAlgorithm, WorldModel, SearchConfig, 
                                                State, Action, Example, Trace)

class MCTSNode(Generic[State, Action, Example]):
    id_iter = itertools.count()

    @classmethod
    def reset_id(cls):
        cls.id_iter = itertools.count()

    def __init__(self, 
                 state: Optional[State], 
                 action: Optional[Action], 
                 parent: "Optional[MCTSNode]" = None,
                 fast_reward: float = 0., 
                 fast_reward_details=None,
                 is_terminal: bool = False, 
                 calc_q: Callable[[list[float]], float] = np.mean
                 ) -> None:
        """
        A node in the MCTS search tree

        :param state: the current state
        :param action: the action of the last step, i.e., the action from parent node to current node
        :param parent: the parent node, None if root of the tree
        :param fast_reward: an estimation of the reward of the last step
        :param is_terminal: whether the current state is a terminal state
        :param calc_q: the way to calculate the Q value from histories. Defaults: np.mean
        """
        self.id = next(MCTSNode.id_iter)
        if fast_reward_details is None:
            fast_reward_details = {}
        self.cum_rewards: list[float] = []
        self.fast_reward = self.reward = fast_reward
        self.fast_reward_details = fast_reward_details
        self.is_terminal = is_terminal
        self.action = action
        self.state = state
        self.parent = parent
        self.children: 'Optional[list[MCTSNode]]' = None
        self.calc_q = calc_q
        if parent is None:
            self.depth = 0
        else:
            self.depth = parent.depth + 1

    # noinspection PyPep8Naming
    @property
    def Q(self) -> float:
        if self.state is None:
            return self.fast_reward
        else:
            return self.calc_q(self.cum_rewards)
        
    def __str__(self) -> str:
        # Using rich to capture formatted string for __str__
        console = Console(file=StringIO(), width=60)
        table = Table(title=f"Node ID: {self.id}", show_header=True, header_style="bold cyan")
        
        table.add_column("Attribute", style="dim")
        table.add_column("Value")
        
        table.add_row("State", str(self.state))
        table.add_row("Action", str(self.action))
        table.add_row("Parent ID", str(self.parent.id if self.parent else "None"))
        table.add_row("Q-Value", f"{self.Q:.2f}")
        
        console.print(table)
        return console.file.getvalue()

    def __repr__(self) -> str:
        # Concise representation for __repr__
        return f"MCTSNode(id={self.id}, state={self.state}, action={self.action}, Q={self.Q:.2f}), num_children={len(self.children)}"


class MCTSResult(NamedTuple):
    """ Simple Container Class for MCTS Output """
    
    terminal_state: State
    cum_reward: float
    trace: Trace
    trace_of_nodes: list[MCTSNode]
    tree_state: MCTSNode
    trace_in_each_iter: list[list[MCTSNode]] = None
    tree_state_after_each_iter: list[MCTSNode] = None
    aggregated_result: Optional[Hashable] = None


class MCTSAggregation(Generic[State, Action, Example], ABC):
    def __init__(self, 
                 retrieve_answer: Callable[[State], Hashable],
                 weight_policy: str = 'edge'
                 ):
        """ 
        Aggregates the results from a MCTS tree. Primary function is to traverse the search tree 
        and aggregate results based on defined policy 
        
        :retrieve_answer: A callable function that retrieves an answer from a node's state. This answer is aggregated across the tree
        
        :weight_policy: A string that defines how to weight the results when aggregating them
        """
        
        
        assert weight_policy in ['edge', 'edge_inverse_depth', 'uniform']
        self.retrieve_answer = retrieve_answer
        self.weight_policy = weight_policy

    def __call__(self, tree_state: MCTSNode[State, Action,Example]) -> Optional[Hashable]:
        answer_dict = defaultdict(lambda: 0)

        def visit(cur: MCTSNode[State, Action, Example]):
            if cur.state is None:
                return []
            if cur.is_terminal:
                answer = self.retrieve_answer(cur.state)
                if answer is None:
                    print("MCTSAggregation: no answer retrieved.")
                    return []
                if self.weight_policy == 'edge':
                    answer_dict[answer] += cur.reward
                elif self.weight_policy == 'edge_inverse_depth':
                    answer_dict[answer] += cur.reward / cur.depth
                elif self.weight_policy == 'uniform':
                    answer_dict[answer] += 1.0
                return [(answer, cur.depth)]
            depth_list = defaultdict(list)
            cur_list = []
            for child in cur.children:
                cur_list.extend(child_info := visit(child))
                for answer, depth in child_info:
                    depth_list[answer].append(depth)
            for answer, depths in depth_list.items():
                if self.weight_policy == 'edge':
                    answer_dict[answer] += cur.reward
                elif self.weight_policy == 'edge_inverse_depth':
                    answer_dict[answer] += cur.reward / np.mean(depths)
            return cur_list

        visit(tree_state)

        if len(answer_dict) == 0:
            return None
        
        return max(answer_dict, key=lambda answer: answer_dict[answer])

class MCTS(SearchAlgorithm, Generic[State, Action, Example]):
    def __init__(self,
                 output_trace_in_each_iter: bool = False,
                 w_exp: float = 1.,
                 depth_limit: int = 5,
                 n_iters: int = 10,
                 cum_reward: Callable[[list[float]], float] = sum,
                 calc_q: Callable[[list[float]], float] = np.mean,
                 simulate_strategy: str | Callable[[list[float]], int] = 'max',
                 output_strategy: str = 'max_reward',
                 uct_with_fast_reward: bool = True,
                 aggregator: Optional[MCTSAggregation] = None,
                 disable_tqdm: bool = True,
                 node_visualizer: Callable[[MCTSNode], dict] = lambda x: x.__dict__
                 ) -> None:
        """
        MCTS algorithm

        :param output_trace_in_each_iter: whether to output the trace of the chosen trajectory in each iteration ; the trace is *deepcopy*-ed
                                          will also output *tree_state_after_each_iter*, which is the *deepcopy*-ed root
        :param w_exp: the weight of exploration in UCT
        :param cum_reward: the way to calculate the cumulative reward from each step. Defaults: sum
        :param calc_q: the way to calculate the Q value from histories. Defaults: np.mean
        :param simulate_strategy: simulate strategy. Options: 'max', 'sample', 'random', or use a custom function
        :param output_strategy: the way to output the result. The nodes are not *deepcopy*-ed, so the information is after all iterations
                                Options: 'max_reward': dfs on the final tree to find a trajectory with max reward using :param cum_reward:
                                         'follow_max': starting from root, choose the maximum reward child at each step. May output a non-terminal node if dead end
                                         'max_visit': the terminal node with maximum number of visits
                                         'max_iter': the trajectory with a terminal node and max reward among those in each iteration
                                         'last_iter': the last trajectory. May output a non-terminal node if the last iteration leads to a dead end
                                         'last_terminal_iter': the last trajectory with a terminal node
                                Outputs *None* if no trajectory with terminal node but required
        :param uct_with_fast_reward: if True, use fast_reward instead of reward for unvisited children in UCT
                                     Otherwise, visit the *unvisited* children with maximum fast_reward first
        """
        super().__init__()
        self.world_model = None
        self.search_config = None
        self.output_trace_in_each_iter = output_trace_in_each_iter
        self.w_exp = w_exp
        self.depth_limit = depth_limit
        self.n_iters = n_iters
        self.cum_reward = cum_reward
        self.calc_q = calc_q
        default_simulate_strategies: dict[str, Callable[[list[float]], int]] = {
            'max': lambda x: np.argmax(x),
            'sample': lambda x: np.random.choice(len(x), p=x),
            'random': lambda x: np.random.choice(len(x)),
        }
        self.simulate_choice: Callable[[list[float]], int] = default_simulate_strategies.get(simulate_strategy,
                                                                                             simulate_strategy)
        assert output_strategy in ['max_reward', 'follow_max', 'max_visit', 'max_iter', 'last_iter',
                                   'last_terminal_iter']
        self.output_strategy = output_strategy
        self.uct_with_fast_reward = uct_with_fast_reward
        self._output_iter: list[MCTSNode] = None
        self._output_cum_reward = -math.inf
        self.trace_in_each_iter: list[list[MCTSNode]] = None
        self.root: Optional[MCTSNode] = None
        self.disable_tqdm = disable_tqdm
        self.node_visualizer = node_visualizer
        self.aggregator = aggregator

    def iterate(self, node: MCTSNode) -> list[MCTSNode]:
        """ Runs a single iteration of MCTS on a given node and returns the path of nodes 
        
        Inputs
        ======
        :node: Root Node to run on 
        
        Outputs: 
        =======
        :path: List of Nodes that corresponds to the optimal path
        """
        
        path = self._select(node) # select a path from root node down to leaf node or not that is not fully expanded

        if not self._is_terminal_with_depth_limit(path[-1]): # if last node is not terminal 
            self._expand(path[-1]) # expand on last node --> make all of the children 
            self._simulate(path) # simulate the path
            
        cum_reward = self._back_propagate(path)
        
        if self.output_strategy == 'max_iter' and path[-1].is_terminal and cum_reward > self._output_cum_reward:
            self._output_cum_reward = cum_reward
            self._output_iter = path
        if self.output_strategy == 'last_iter':
            self._output_cum_reward = cum_reward
            self._output_iter = path
        if self.output_strategy == 'last_terminal_iter' and path[-1].is_terminal:
            self._output_cum_reward = cum_reward
            self._output_iter = path
            
        return path

    def _is_terminal_with_depth_limit(self, node: MCTSNode) -> bool:
        """ Returns bool if node is terminal or if (depth of the tree >= the limit) """
        return node.is_terminal or node.depth >= self.depth_limit

    def _select(self, node: MCTSNode) -> list[MCTSNode]:
        """ 
        Traverses tree from input node via selecting max UCT child until terminal, 
        then returns list of nodes representing path
        """
        path = []
        while True:
            path.append(node)
            # if node has no children, return the path  
            if node.children is None or len(node.children) == 0 or self._is_terminal_with_depth_limit(node):
                return path
            # else, the next node is the child node with best UCT
            node = self._uct_select(node)

    def _uct_select(self, node: MCTSNode) -> MCTSNode:
        """ 
        Supposing the node is fully expanded (aka max children), selects and returns the best child node (maxes UCT) out of the children 
        
        :node: the current node you are at in your tree search
        
        Note - This is called recursively in "_select" as you traverse the tree
        """
        # all children visited for selecting best one 
        if self.uct_with_fast_reward or all(child.state is not None for child in node.children):
            # remember, self._uct(child) is applied to each element in node.children, then the max is returned
            return max(node.children, key=self._uct) # finds the max 
        # filter for unvisited children, then get the max fast reward
        else:
            unvisited_children = filter(lambda child: child.state is None, node.children)
            
            return max(unvisited_children, key=lambda child: child.fast_reward)
        
    def _uct(self, node: MCTSNode) -> float:
        """ Gets the current UCT value for the node """
        return node.Q + self.w_exp * np.sqrt(np.log(len(node.parent.cum_rewards)) 
                                             / max(1, len(node.cum_rewards)))

    def _expand(self, node: MCTSNode) -> None:
        """ 
        Expand the node to make all its children
        
        Note - Does not return anything, it just sets the attributes of the current node
        """
        # if current ndoe is
        if node.state is None:
            node.state, aux = self.world_model.step(node.parent.state, node.action)
            # reward is calculated after the state is updated, so that the
            # information can be cached and passed from the world model
            # to the reward function with **aux without repetitive computation
            node.reward, node.reward_details = self.search_config.reward(node.parent.state, 
                                                                         node.action, 
                                                                         **node.fast_reward_details, 
                                                                         **aux)
            node.is_terminal = self.world_model.is_terminal(node.state)

        # if node is terminal, just return
        if node.is_terminal:
            return

        # else, get action space
        children = []
        actions = self.search_config.get_actions(node.state)
        # for each action, make a child and set the action used to get there
        for action in actions:
            fast_reward, fast_reward_details = self.search_config.fast_reward(node.state, action)
            # 
            child = MCTSNode(state=None, action=action, parent=node,
                             fast_reward=fast_reward, fast_reward_details=fast_reward_details, calc_q=self.calc_q)
            children.append(child)

        node.children = children

    def _simulate(self, path: list[MCTSNode]):
        node = path[-1]
        while True:
            if node.state is None:
                self._expand(node)
            if self._is_terminal_with_depth_limit(node) or len(node.children) == 0:
                return
            fast_rewards = [child.fast_reward for child in node.children]
            node = node.children[self.simulate_choice(fast_rewards)]
            path.append(node)

    def _back_propagate(self, path: list[MCTSNode]):
        rewards = []
        cum_reward = -math.inf
        for node in reversed(path):
            rewards.append(node.reward)
            cum_reward = self.cum_reward(rewards[::-1])
            node.cum_rewards.append(cum_reward)
        return cum_reward

    def _dfs_max_reward(self, path: list[MCTSNode]) -> tuple[float, list[MCTSNode]]:
        cur = path[-1]
        if cur.is_terminal:
            return self.cum_reward([node.reward for node in path[1:]]), path
        if cur.children is None:
            return -math.inf, path
        visited_children = [x for x in cur.children if x.state is not None]
        if len(visited_children) == 0:
            return -math.inf, path
        return max((self._dfs_max_reward(path + [child]) for child in visited_children), key=lambda x: x[0])

    def search(self):
        self._output_cum_reward = -math.inf 
        self._output_iter = None
        self.root = MCTSNode(state=self.world_model.init_state(), action=None, parent=None, calc_q=self.calc_q)
        if self.output_trace_in_each_iter:
            self.trace_in_each_iter = []

        for _ in trange(self.n_iters, disable=self.disable_tqdm, desc='MCTS iteration', leave=False):
            path = self.iterate(self.root)
            if self.output_trace_in_each_iter:
                self.trace_in_each_iter.append(deepcopy(path))

        if self.output_strategy == 'follow_max':
            self._output_iter = []
            cur = self.root
            while True:
                self._output_iter.append(cur)
                if cur.is_terminal:
                    break
                visited_children = [x for x in cur.children if x.state is not None]
                if len(visited_children) == 0:
                    break
                cur = max(visited_children, key=lambda x: x.reward)
            self._output_cum_reward = self.cum_reward([node.reward for node in self._output_iter[1::-1]])
        if self.output_strategy == 'max_reward':
            self._output_cum_reward, self._output_iter = self._dfs_max_reward([self.root])
            if self._output_cum_reward == -math.inf:
                self._output_iter = None

    def __call__(self,
                 world_model: WorldModel[State, Action, Example],
                 search_config: SearchConfig[State, Action, Example],
                 log_file: Optional[str] = None,
                 **kwargs
                 ) -> MCTSResult:
        MCTSNode.reset_id()
        self.world_model = world_model
        self.search_config = search_config

        self.search()

        if self._output_iter is None:
            terminal_state = trace = None
        else:
            terminal_state = self._output_iter[-1].state
            trace = [node.state for node in self._output_iter], [node.action for node in self._output_iter[1:]]
        if self.output_trace_in_each_iter:
            trace_in_each_iter = self.trace_in_each_iter
            tree_state_after_each_iter = [trace[0] for trace in trace_in_each_iter]
        else:
            trace_in_each_iter = tree_state_after_each_iter = None
        result = MCTSResult(terminal_state=terminal_state,
                            cum_reward=self._output_cum_reward,
                            trace=trace,
                            trace_of_nodes=self._output_iter,
                            tree_state=self.root,
                            trace_in_each_iter=trace_in_each_iter,
                            tree_state_after_each_iter=tree_state_after_each_iter)
        if self.aggregator is not None:
            result = MCTSResult(
                terminal_state=result.terminal_state,
                cum_reward=result.cum_reward,
                trace=result.trace,
                trace_of_nodes=result.trace_of_nodes,
                tree_state=result.tree_state,
                trace_in_each_iter=result.trace_in_each_iter,
                tree_state_after_each_iter=result.tree_state_after_each_iter,
                aggregated_result=self.aggregator(result.tree_state),
            )
        return result
    
    