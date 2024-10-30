from __future__ import annotations

""" Tests with MCTS Node """
import pickle
from os import PathLike
import pickle
import math
from copy import deepcopy
from typing import Generic, Optional, NamedTuple, Callable, Hashable, Any, Literal, Tuple
import itertools
from abc import ABC
from collections import defaultdict
from rich.console import Console
from rich.table import Table
from io import StringIO
import numpy as np
from tqdm import trange
import gymnasium as gym
import random
from textwrap import dedent
from rich.tree import Tree

from agents.environments.game import GymGame
from agents.algorithms.tree_search.base import (SearchAlgorithm, WorldModel, SearchConfig, 
                                                State, Action, Example, Trace)
from agents.utils import calculate_returns
from agents.algorithms.tree_search.mcts_simple import MCTS, MCTSNode
from agents.environments.game import DiscreteGymGame

import gymnasium as gym 
import ale_py

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
        
        Inputs:
        ======
        :param state: the current state
        :param action: the action of the last step, i.e., the action from parent node to current node
        :param parent: the parent node, None if root of the tree
        :param is_terminal: whether the current state is a terminal state
        :param calc_q: the way to calculate the Q value from histories. Defaults: np.mean
        
        Internal: 
        =========
        :param cum_rewards: stores the cumulative rewards in each iteration of mcts ie. [tot_rewards_iter1, iter2, ]
        :param reward: the one-off reward of the node during one iteration of mcts ie [100] from rollout in iter1
        :param children: contains the children nodes
        :param depth: depth of the node in the tree
        
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
        """ 
        My Q function: 
        Note - if self.cum_rewards = [] aka it has not been visited before, 
        make Q value --> inf aka SUPER exploration (each child node visited at least once)
        
        """
        if self.cum_rewards: # if node been explored before 
            return self.calc_q(self.cum_rewards)
        else: # else, Q value to infinity --> explore
            return np.inf
        
    def __str__(self) -> str:
        # Using rich to capture formatted string for __str__
        console = Console(file=StringIO(), width=60)
        table = Table(title=f"Node ID: {self.id}", show_header=True, header_style="bold cyan")
        
        table.add_column("Attribute", style="dim")
        table.add_column("Value")
        
        table.add_row("State", str(self.state.shape))
        table.add_row("Action", str(self.action))
        table.add_row("Parent ID", str(self.parent.id if self.parent else "None"))
        table.add_row("Q-Value", f"{self.Q:.2f}")
        # table.add_row("Reward", f"{self.Q:.2f}")
        table.add_row("Number Children", f"{len(self.children)}")
        console.print(table)
        return console.file.getvalue()

    def __repr__(self) -> str:
        # Concise representation for __repr__
        return dedent(f"""
    MCTSNode(id={self.id}, 
    state=dim:{self.state.shape}, 
    action={self.action}, 
    Q={self.Q:.2f}, 
    reward={self.reward}, 
    num_children={len(self.children)})
    """)

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
        
    def _is_terminal_with_depth_limit(self, node: MCTSNode) -> bool:
        """ True if node is terminal or depth limit exceeded """
        return bool(node.is_terminal or node.depth >= self.depth_limit)
    
    def _uct(self, node: MCTSNode) -> float:
        """ Gets the current UCT value for the node """
        N = len(node.parent.cum_rewards) # num times parent node visited
        n_i = max(1, len(node.cum_rewards)) # num times child node visited 
        term = self.w_exp * np.sqrt(np.log(N) / n_i)
        
        return node.Q + term
    
    def _uct_select(self, node: MCTSNode) -> MCTSNode:
        """ 
        Supposing the node is fully expanded (aka max children), selects and returns the best child node (maxes UCT) out of the children 
        
        :node: the current node you are at in your tree search
        
        Note - This is called recursively in "_select" as you traverse the tree
        Note - no fast reward, so node must be fully expanded
        """
        return max(node.children, key=self._uct)
    
    def select(self, node: MCTSNode) -> list[MCTSNode]: 
        ''' Goes through start node, and traverses via selecting best uct of children. If no children, return path as is '''
        path = []
        while True: 
            path.append(node)
            if self._is_terminal_with_depth_limit(node) or len(node.children)==0: 
                return path # NOTE: return if ...
            best_child = self._uct_select(node)
            node = best_child # set node as best child
    
    def expand(self, node: MCTSNode, environment: GymGame | gym.Env, num_children: int) -> None: 
        """ Expands last node of path into d children and updates the nodes internal children attribute """
        action_space = list(range(environment.action_cardinality))
        actions:list[int] = list(random.sample(action_space, num_children))
        # TODO: make a reset-able environment for gym.Env classes
        envs:list[GymGame] = [environment.get_copy() for _ in actions]
        
        children = []
        for action, env in zip(actions, envs): 
            next_obs, reward, terminated, *_ = env.step(action)
            child = MCTSNode(state=next_obs, 
                             action=action, 
                             reward=None, # NOTE: we do not add reward to this node for now
                             parent=node, 
                             is_terminal=terminated, 
                             calc_q=self.calc_q)
            children.append(child)
        node.children = children # update leaf/terminal node with children 
        
    def _single_rollout(self, 
                        node: MCTSNode, 
                        environment: gym.Env | GymGame, 
                        strategy: Literal['random', 'policy']='random', 
                        max_tries: int = 10
                        ) -> float | int: 
        """ Runs a full rollout on a single child node and returns the summed cumulative rewards from rollout """
        def episode_stop_condition(step: int, terminated: bool) -> bool: 
            """ True if max_tries exceeded or terminal"""
            return bool(step > max_tries or terminated)
        
        if isinstance(environment, GymGame): 
            action_space = list(range(environment.action_cardinality))
        else: 
            action_space = list(range(environment.action_space.n)) 
            
        step = 0
        terminated = False
        rewards = []
        while not episode_stop_condition(step, terminated):
            if strategy == 'random': 
                action = random.sample(action_space, 1)[0]
            elif strategy == 'policy': 
                action = ...
            next_obs, reward, terminated, *_ = environment.step(action)
            rewards.append(reward)
            obs = next_obs
        # node.reward = sum(rewards) # make node.reward = sum of rewards collected during the rollout
        return sum(rewards)
        
    def simulate(self, 
                 path: list[MCTSNode],
                 environment: gym.Env,
                 strategy: Literal['random', 'policy']='random', 
                 max_tries: int = 100
                 ) -> MCTSNode: 
        """ 
        Perform a rollout on the selected child node, and return the simulated node with update rollout reward
        
        Notes:
        =====
        1 <= d <= cardinality-action_space 
        """ 
        child_idx: int = random.sample(range(len(path[-1].children)), 1)[0]
        node_to_sim: MCTSNode = path[-1].children[child_idx] # for now, just first node added
        
        rollout_args: dict = {
        'node': node_to_sim, 
        'environment': environment, 
        'strategy': strategy, 
        'max_tries': max_tries
        }
        
        rollout_reward: float | int = self._single_rollout(**rollout_args)
        node = path[-1].children[child_idx]
        node.reward = rollout_reward
        path.append(node)
        
    
    def back_propagate(self, path: list[MCTSNode]) -> float:
        """ 
        Updates each node in the path with the cumulative rewards from rollout and returns the updated path and the cum_reward for the root 
        
        ex. leaf node gets rollout reward 
        leaf node - 1 gets rollout reward + own reward 
        leaf node - 2 gets rollout reward + leaf node -1 + own reward 
        ...
        
        :param path - list[MCTSNode]: list of nodes corresponding to search path 
        :param child_idx - int: Inside the leaf node, the idx of its expanded child node we simulated
        """
        cum_reward_func: Callable[[list[float]], float] = self.cum_reward # way to calculate the cumulative reward from each step. Defaults: sum
        rewards = [] # holds rewards for each node
        cum_reward = -math.inf
        for node in reversed(path): # leaf --> root
            rewards.append(node.reward) # ex. leaf: rewards = [100]; leaf-1: rewards = [100, 10]; leaf-2: rewards = [100, 10, 15], ...
            # NOTE: work-around for node.reward = None => we filter this out 
            rewards = list(filter(lambda x: x != None, rewards))
            cum_reward = cum_reward_func(rewards[::-1]) # self.cum_rewards callable sum; ex. sum([10, 100]), sum([15, 10, 100])
            node.cum_rewards.append(cum_reward) # node.cum_rewards stores summed rewards for one iteration; ex. (leaf-1).cum_reward = 110

        return cum_reward
    
    def iterate(self, 
                node: MCTSNode, 
                environment: GymGame | gym.Env, 
                num_children: int = 6,  
                strategy: Literal['random', 'policy']='random', 
                max_tries: int = 10
                ) -> list[MCTSNode]:
        """ Runs a single iteration of MCTS on a given node and returns the path of nodes 
        
        Inputs
        ======
        :node: Root Node to run on 
        
        Outputs: 
        =======
        :path: List of Nodes that corresponds to the optimal path
        """
        
        path = self.select(node) # select a path from root node down to leaf node or not that is not fully expanded
        if not self._is_terminal_with_depth_limit(path[-1]): # if last node is not terminal 
            self.expand(path[-1], environment, num_children) # expand on last node --> make all of the children 
            self.simulate(path, environment, strategy, max_tries) # simulate the path

        cum_reward = self.back_propagate(path)
        _, _ = environment.reset()
        
        if self.output_strategy == 'max_iter' and path[-1].is_terminal and cum_reward > self._output_cum_reward:
            self._output_cum_reward = cum_reward
            self._output_iter = path
        elif self.output_strategy == 'last_iter':
            self._output_cum_reward = cum_reward
            self._output_iter = path
        elif self.output_strategy == 'last_terminal_iter' and path[-1].is_terminal:
            self._output_cum_reward = cum_reward
            self._output_iter = path
        
        return path
    
    def display_path(self, path: list[MCTSNode]) -> None:
        console = Console()
        tree = Tree(f"[bold green]MCTS Path - Depth: {len(path)}[/]")
        for node in path:
            children = node.children if node.children else "None"
            node_info = f"""
            Node ID: {node.id} | State: {node.state.shape} | Action: {node.action} | Q-Value: {node.Q:.2f} | Parent: {node.parent} | Children: {children}
            """
            tree.add(node_info)
        console.print(tree)

if __name__=="__main__": 
    
    gym.register_envs(ale_py)
    env = gym.make("ALE/AirRaid-v5")
    env = DiscreteGymGame(env)
    A = env.action_cardinality
    obs, info = env.reset()
    
    root = MCTSNode(state=obs, action=None)
    mcts = MCTS()
    
    for _ in range(2): 
        path = mcts.iterate(root, env)
    
    mcts.display_path(path)
    breakpoint()

    # opt_path: list[MCTSNode] = mcts.select(root)
    # opt_path[-1] = mcts.expand(opt_path[-1], env, A) # expand the leaf node 
    # node_idx = random.sample(range(len(opt_path[-1].children)), 1)[0]
    # breakpoint()
    # simulated_child: MCTSNode = mcts.simulate(opt_path[-1].children[node_idx], env, 'random', max_tries=10)
    # opt_path[-1].children[node_idx] = simulated_child
    # breakpoint()
    # opt_path, cum_reward = mcts.back_propagate(opt_path, node_idx)
    # breakpoint()
    
    
    