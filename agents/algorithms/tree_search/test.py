from __future__ import annotations

""" Tests with MCTS Node """
import pickle
from os import PathLike
import pickle
import math
from copy import deepcopy
from typing import Generic, Optional, NamedTuple, Callable, Hashable, Any, Literal
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
from agents.environments.game import GymGame
from agents.algorithms.tree_search.base import (SearchAlgorithm, WorldModel, SearchConfig, 
                                                State, Action, Example, Trace)
# from agents.algorithms.tree_search.mcts import MCTSNode, MCTSAggregation, MCTS

from agents.algorithms.tree_search.mcts_simple import MCTS, MCTSNode
from agents.environments.game import DiscreteGymGame

import gymnasium as gym 
import ale_py

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
        ''' Goes through start node, and traverses via selecting best uct of children '''
        path = []
        while not self._is_terminal_with_depth_limit(node): 
            path.append(node)
            best_child = self._uct_select(node)
            node = best_child # set node as best child
        return path
    
    def expand(self, node: MCTSNode, environment: gym.Env, num_children: int) -> MCTSNode: 
        """ Expands last node of path into d children """
        action_space = list(range(environment.action_space.n))
        actions:list[int] = list(random.sample(action_space, num_children))
        envs:list[GymGame] = [GymGame.get_copy() for _ in actions]
        
        children = []
        for action, env in zip(actions, envs): 
            next_obs, reward, terminated, _, _ = env.step(action)
            child = MCTSNode(state=next_obs, 
                             action=action, 
                             reward=reward, 
                             parent=node, 
                             is_terminal=terminated, 
                             calc_q=self.calc_q)
            children.append(child)
            
        node.children = children
        
        return node
    
    def _single_rollout(self, 
                        node: MCTSNode, 
                        environment: gym.Env, 
                        strategy: Literal['random', 'policy']='random', 
                        max_tries: int = 10
                        ): 
        """ Runs a full rollout on a single child node """
        def episode_stop_condition(step: int, terminated: bool) -> bool: 
            return bool(step > max_tries or terminated)
        
        action_space = list(range(environment.action_space.n)) 
        step = 0
        terminated = False
        rewards = []
        while not episode_stop_condition(step, terminated):
            if strategy is 'random': 
                action = random.sample(action_space, 1)
            elif strategy is 'policy': 
                action = ...
                
            next_obs, reward, terminated, _, _ = environment.step(action)
            
            rewards.append(reward)
            obs = next_obs
            
        node.cum_rewards = rewards
        
        return node
        
    def simulate(self, 
                 nodes: MCTSNode | list[MCTSNode], 
                 environment: gym.Env,
                 strategy: Literal['random', 'policy']='random', 
                 max_tries: int = 100
                 ) -> list[MCTSNode]: 
        """ Perform rollout on d nodes, 1 <= d <= cardinality-action_space """ 
        if isinstance(nodes, list): 
            nodes = [MCTSNode]
            
        rollouts_args:list[dict] = [{
            'node': node, 
            'environment': environment, 
            'strategy': strategy, 
            'max_tries': max_tries
            } for node in nodes]
            
        simulated_nodes = [self._single_rollout(**arg) for arg in rollouts_args]
            
        return simulated_nodes
    
    def backpropagate(self) -> None: 
        return ...
    
    def iterate(self): 
        
        return ...

if __name__=="__main__": 
    
    gym.register_envs(ale_py)
    env = gym.make("ALE/AirRaid-v5")
    env = DiscreteGymGame(env)
    action_space_cardinality = len(env.env.action_space.n)
    obs, info = env.reset()
    
    root = MCTSNode(state=obs, action=None)
    mcts = MCTS()
    
    breakpoint()
    opt_path = mcts.select(root)
    opt_path[-1] = mcts.expand(opt_path[-1], env, action_space_cardinality) # expand the leaf node 
    opt_path[-1] = mcts.simulate(opt_path[-1], env, 'random', max_tries=10)
    
    