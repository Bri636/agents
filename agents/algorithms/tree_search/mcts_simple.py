""" Credit to: https://github.com/maitrix-org/llm-reasoners """
from __future__ import annotations

import pickle
from os import PathLike
import pickle
import math
from copy import deepcopy
from typing import Generic, Optional, NamedTuple, Callable, Hashable, Any
import itertools
from abc import ABC
from collections import defaultdict
from rich.console import Console
from rich.table import Table
from io import StringIO
import numpy as np
from tqdm import trange
import gymnasium as gym

from agents.environments.game import GymGame
from agents.algorithms.tree_search.base import (SearchAlgorithm, WorldModel, SearchConfig, 
                                                State, Action, Example, Trace)

class MCTSNode(Generic[State, Action, Example]):
    
    id_iter = itertools.count() # iterator; each next returns next step as int

    @classmethod
    def reset_id(cls):
        cls.id_iter = itertools.count() 

    def __init__(self, 
                 state: Optional[State], 
                 action: Optional[Action], 
                 reward: float = None,
                 parent: "Optional[MCTSNode]" = None,
                 is_terminal: bool = False, 
                 calc_q: Callable[[list[float]], float] = np.mean
                 ) -> None:
        """
        A node in the MCTS search tree

        :param state: the current state
        :param action: the action of the last step, i.e., the action from parent node to current node
        :param parent: the parent node, None if root of the tree
        :param is_terminal: whether the current state is a terminal state
        :param calc_q: the way to calculate the Q value from histories. Defaults: np.mean
        
        Note - Action_(t-1), State_(t), and Reward_(t) per Node
        Note - root: Action = None, State_(0), Reward_(0) = None
        """
        self.id = next(MCTSNode.id_iter)
        
        self.cum_rewards: list[float] = [] # rewards from rollout
        self.is_terminal = is_terminal
        self.action = action
        self.state = state
        self.reward = reward
        self.parent = parent
        self.children: 'Optional[list[MCTSNode]]' = []
        self.calc_q = calc_q
        
        # if no parent => root 
        if parent is None:
            self.depth = 0
        else:
            self.depth = parent.depth + 1 

    # noinspection PyPep8Naming
    @property
    def Q(self) -> float:
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
        return f"MCTSNode(id={self.id}, state=dim:{self.state.shape}, action={self.action}, Q={self.Q:.2f}), num_children={len(self.children)}"
    
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
        
        Note - Since no fast_reward instead of reward for unvisited children in UCT we HAVE to visit the *unvisited* children with maximum fast_reward first
        """
        super().__init__()
        default_simulate_strategies: dict[str, Callable[[list[float]], int]] = {
            'max': lambda x: np.argmax(x),
            'sample': lambda x: np.random.choice(len(x), p=x),
            'random': lambda x: np.random.choice(len(x)),
        }
        self.simulate_choice: Callable[[list[float]], int] = default_simulate_strategies.get(simulate_strategy,
                                                                                        simulate_strategy)
        # self.world_model = None
        # self.search_config = None
        self.output_trace_in_each_iter = output_trace_in_each_iter
        self.w_exp = w_exp
        self.depth_limit = depth_limit
        self.n_iters = n_iters
        self.cum_reward = cum_reward
        self.calc_q = calc_q
        assert output_strategy in ['max_reward', 'follow_max', 
                                   'max_visit', 'max_iter', 
                                   'last_iter', 'last_terminal_iter']
        self.output_strategy = output_strategy
        self._output_iter: list[MCTSNode] = None
        self._output_cum_reward = -math.inf
        self.trace_in_each_iter: list[list[MCTSNode]] = None
        self.root: Optional[MCTSNode] = None
        self.disable_tqdm = disable_tqdm
        self.node_visualizer = node_visualizer

    def iterate(self, node: MCTSNode) -> list[MCTSNode]:
        """ Runs a single iteration of MCTS on a given node and returns the path of nodes 
        
        Inputs
        ======
        :node: Root Node to run on 
        
        Outputs: 
        =======
        :path: List of Nodes that corresponds to the optimal path
        """
        path = self._select_path(node) # select a path from root node down to leaf node or not that is not fully expanded
        last_node = path[-1]
        if not self._is_terminal_with_depth_limit(last_node): # if last node is not terminal 
            self._expand_node(last_node) # expand on last node --> make all of the children 
            self._simulate(path) # simulate the path rollouts
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

    def _select_path(self, node: MCTSNode) -> list[MCTSNode]:
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
        Note - no fast reward, so node must be fully expanded
        """
        return max(node.children, key=self._uct) # finds the max UCT of the children 
        
    def _uct(self, node: MCTSNode) -> float:
        """ Gets the current UCT value for the node """
        return node.Q + self.w_exp * np.sqrt(np.log(len(node.parent.cum_rewards)) 
                                             / max(1, len(node.cum_rewards)))

    def _expand_node(self, node: MCTSNode, environment: gym.Env | GymGame) -> None:
        """ 
        Expand the node to make all its children - uses reward
        Note - every node passed has a state, so no fast rewards
        Note - Does not return anything, it just sets the attributes of the current node
        """
        # if node is terminal, just return
        if node.is_terminal:
            return
        # else, get action space
        children = []
        # actions = self.search_config.get_actions(node.state)
        if isinstance(environment, GymGame): 
            action_space = environment.legal_actions()
        else: 
            action_space = [i for i in range(environment.action_space.n)]
        sub_copied_envs: list[gym.Env] = [environment.get_copy() for _ in action_space]
        
        # for each action, make a child and set the action used to get there
        for action, env in zip(action_space, sub_copied_envs):
            # fast_reward, fast_reward_details = self.search_config.fast_reward(node.state, action)
            obs, reward, terminated, *_ = env.step(action)
            child = MCTSNode(state=obs, 
                             action=action, 
                             reward=reward,
                             parent=node,
                             calc_q=self.calc_q, 
                             is_terminal=terminated)
            children.append(child)

        node.children = children # set input node's children as result of executing all action_space on input node

    def _simulate(self, path: list[MCTSNode]):
        """ Goes through all of the children """
        node = path[-1]
        while True: # keep going 
            if node.state is None:
                self._expand_node(node)
            if self._is_terminal_with_depth_limit(node) or len(node.children) == 0: # if node is terminal or at depth limit or no children, 
                return
            child_rewards = [child.reward for child in node.children]
            node = node.children[self.simulate_choice(child_rewards)] # select child with best reward 
            
            path.append(node) # add to path then iterate again 

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
        
        return result