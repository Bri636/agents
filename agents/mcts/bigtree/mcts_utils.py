""" Util classes used in MCTS, including reward strategies and search strategies """

from __future__ import annotations
from typing import Tuple, Callable, TypeVar, Optional
import math 
import functools

from agents.generators.base_generator import BaseLLMGenerator
from agents.mcts.bigtree.bigtree_mcts_node import BTMCTSNode

T = TypeVar('T')

def register_strategy(strategy_dict, name=None):
    """Decorator to register a method as a search strategy."""
    def decorator(func):
        strategy_name = name or func.__name__
        strategy_dict[strategy_name] = func
        return func
    return decorator

class SearchStrategies: 
    """ Bundle of Search Strategies for traversing MCTS Tree """
    strategies = {} # storage 

    @register_strategy(strategies, name='follow_max')
    def greedy_next(self, root: BTMCTSNode, cum_reward_func: Callable) -> Tuple[list[BTMCTSNode], float]: 
        """ Selects the optimal path based on choosing child with max reward """
        output_iter = []  # stores nodes in path
        cur = root  # make node
        while True:
            output_iter.append(cur)  # add to path iteratively
            if cur.is_terminal:  # return path if terminal node
                break
            # get the children from node if child.state is not None
            visited_children = [child for child in cur.children if child.state is not None]
            # if node has no children, then return path self._output_iter
            if len(visited_children) == 0:
                break
            # else, get the max child based on its reward
            cur = max(visited_children, key=lambda x: x.reward)
        # set the output_cum_reward as the sum of the node rewards in the output iter trace
        # NOTE: OG had self._output_iter[1::-1] but this takes idx [1, 0]
        output_cum_reward: float = cum_reward_func(
            [node.reward for node in output_iter[::-1]])
        
        return output_iter, output_cum_reward
    
    @register_strategy(strategies, name='max_reward')
    def dfs_max_reward(self, 
                       path: list[BTMCTSNode] | BTMCTSNode, 
                       cum_reward_func: Callable
                       ) -> tuple[list[BTMCTSNode], float]:
        """ Recursively searches for path that maximizes total reward over the path """
        if isinstance(path, BTMCTSNode):
            path = [path]
        cur = path[-1]
        if not cur.children:
            # Leaf node (no children)
            cumulative_reward: float = cum_reward_func(
                [node.reward for node in path[1:]])
            return path, cumulative_reward
        else:
            max_reward = -math.inf
            best_path = path
            for child in cur.children:
                child_path, reward = self.dfs_max_reward(path + [child], cum_reward_func)
                if reward > max_reward:
                    max_reward = reward
                    best_path = child_path
                    
            return best_path, max_reward
        
    @classmethod
    def execute_strategy(cls, 
                         root: "BTMCTSNode", 
                         cum_reward_func: Callable, 
                         strategy: str
                         ) -> tuple[list["BTMCTSNode"], float]:
        """ Interface for executing strategies."""
        strategy_func: Callable = cls.strategies.get(strategy)
        if not strategy_func:
            raise ValueError(f"Strategy '{strategy}' does not exist. Choose from {list(cls.strategies.keys())}")
        instance = cls()
        return strategy_func(instance, root, cum_reward_func) 
    
# class EvaluationStrategies
        
class RewardRolloutStrategies:
    """ Strategies for assigning rewards for terminal and non-terminal phases """ 
    
    strategies = {}
    
    @register_strategy(strategies, name='base')
    def base(win: bool,
             win_reward: float = 100,
             lose_reward: float = -50
             ) -> float:
        """ Classic Reward Shaping for Win-Loss """
        return win_reward if win else lose_reward
    
    # @classmethod
    # def execute_strategy(cls, win: bool, strategy: str) -> tuple[list["BTMCTSNode"], float]:
    #     """ Interface for executing strategies."""
    #     strategy_func: Callable = cls.strategies.get(strategy)
    #     if not strategy_func:
    #         raise ValueError(f"Strategy '{strategy}' does not exist. Choose from {list(cls.strategies.keys())}")
    #     instance = cls()
    #     return strategy_func(instance, root, cum_reward_func) 
    
    
    
class RewardStrategy: 
    """ Strategies for generating """
    strategies = {}
    
    def log_probs(self, answer_prompt, generator, **kwargs): 
        """ Assigns Rewards based on log probs"""
        
    def confidence(self, answer_prompt, generator: BaseLLMGenerator, num_samples: int, **kwargs): 
        """ Perform k samples and then find proportion of most frequent answer """
        
        k_shot_prompts = [answer_prompt] * num_samples
        actions = [generator.generate(prompt) for prompt in k_shot_prompts]
        
        return actions
    
    def mc_rollout(self): 
        """ Do rollout on that """