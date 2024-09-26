from __future__ import annotations

'''UCB implementation of MCTS with BigTree package'''

from typing import Any, Union, Callable
from bigtree import Node, BaseNode
import string
import random
import gymnasium as gym

from agents.algorithms.mcts.base_text import BaseNode, BaseMCTS


class UCBNode(BaseNode): 
    '''Text Node for when rewards are not or given'''
    
    def __init__(self, 
                 state: Any, 
                 action: Any,
                 action_space: Any = None
                 ) -> None:
        super().__init__()
        
        self.state = state
        self.action = action
        self.action_space = action_space
        
        # other attr for tracking node state
        self.children = []
        self.executed_actions = set()
        self.number_visits = 0
        self.value = 0
        
    @property
    def is_fully_expanded(self) -> bool: 
        '''Tests if node is fully expanded by seeing if number children is same as number of actions'''
        return bool(len(self.action_space)==self.children)
    
    @property
    def best_child(self, exploration_weight: float) -> Node: 
        '''Selects the best child node; use after node is fully expanded'''
        
        rand = random.random()
        
        if rand > exploration_weight: # explore 
            return random.sample(self.children, 1)[0] # select node randomly
        else: 
            return max([child_node for child_node in self.children], 
                       key=lambda x: x.value)
            
    def expand(self, action: Any, environment: Union[gym.Environment, Callable], **kwargs) -> None:
        
        for action in self.action_space: 
            
            if action in self.executed_actions: 
                pass
            else: 
                if isinstance(environment, gym.Env): 
                    next_state = environment.step(action)
                elif isinstance(environment, Callable): 
                    next_state = environment(action, **kwargs)
                
            child_node = Node()
            self.children.append(child_node)
        
        return super().expand()
    
    