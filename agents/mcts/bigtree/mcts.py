""" Bigtree version of with LLM MCTS """

from __future__ import annotations
import copy
from typing import Optional, Callable, Any, Literal, Tuple, Generic
from tqdm.rich import trange, tqdm
import random
import math
import logging
import numpy as np

from agents.mcts.bigtree.node import State, Action, Reward, Computable, NodePath, MCTSNode


def win_lose(win: bool,
             win_reward: float = 100,
             lose_reward: float = -50
             ) -> float:
    return win_reward if win else lose_reward


class MCTS(Generic[State, Action]):
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

    def __init__(self,
                 output_trace_in_each_iter: bool = False,
                 w_exp: float = 1.,
                 depth_limit: int = 5,
                 num_iters: int = 10,
                 cum_reward_func: Callable[[Computable], float] = sum,
                 calc_q_func: Callable[[Computable], float] = np.mean,
                 simulate_strategy: str | Callable[[Computable], int] = 'max',
                 output_strategy: str = 'max_reward',
                 use_tqdm: bool = True,
                 reward_strategy: Literal['base'] = 'base',
                 logger: Optional[logging.Logger] = None
                 ) -> None:
        super().__init__()
        rollout_strategies: dict[str, Callable[[list[float]], int]] = {
            'max': lambda x: np.argmax(x),
            'sample': lambda x: np.random.choice(len(x), p=x),
            'random': lambda x: np.random.choice(len(x)),
        }

        reward_strategies = {
            'base': (win_lose, np.mean)  # stored as
        }

        self.simulate_choice: Callable[[list[float]], int] = rollout_strategies.get(simulate_strategy,
                                                                                    simulate_strategy)

        self.output_trace_in_each_iter: bool = output_trace_in_each_iter
        self.w_exp: float = w_exp
        self.depth_limit: int = depth_limit
        self.num_iters: int = num_iters
        self.cum_reward_func: Callable = cum_reward_func
        self.calc_q_func: Callable = calc_q_func
        assert output_strategy in ['max_reward', 'follow_max',
                                   'max_visit', 'max_iter',
                                   'last_iter', 'last_terminal_iter']
        self.output_strategy = output_strategy
        # self._output_iter: list[BTMCTSNode] = None
        # self._output_cum_reward = -math.inf
        self._output_iters: dict[int, NodePath] = {}
        self._output_cum_rewards: dict[int, float] = {}

        self.trace_in_each_iter: dict[int, list[NodePath]] = {}
        self.use_tqdm: bool = use_tqdm
        self.terminal_reward_strategy, self.reward_strategy = reward_strategies.get(
            reward_strategy)

        self.logger = logger

    def select(self, nodes: list[MCTSNode]) -> list[NodePath]:
        """ Selects optimal leaf node for batch of nodes or single node """
        node_paths: list[NodePath] = []
        for node in nodes:
            node_path: NodePath = []
            while not (node.terminal_with_depth_limit(self.depth_limit)
                       and len(node.children) != 0):
                child = node.best_child
                node_path.append(child)
                node = child
            node_paths.append(node_path)
        return nodes

    def expand(self, 
               nodes: list[MCTSNode], 
               actor, 
               world_model, 
               ):
        """ Batch Expands a list of leaf nodes """
        for node in nodes:
            ...
        ...
        
        
if __name__=="__main__": 
    
    
    
    
    breakpoint()
