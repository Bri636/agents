from __future__ import annotations
from typing import Any, TypeVar, Callable, Optional

'''MCTS Tree implementation'''

import random

from bigtree import Node
from bigtree.utils import exceptions
import gymnasium as gym
from torch import nn

from agents.algorithms.mcts_old.base_text import BaseNode
from agents.algorithms.bandits.epsilon import EpsilonDeepQGreedy
from agents.algorithms.mcts_old.q_functions import DeepQImage

T = TypeVar('T')

class DeepQNode:
    '''Node for MCTS with q values'''
    
    def __init__(self,
                 state: Any,
                 action: Any,
                 parent: Any = None,
                 reward: float = 0.0,
                 environment: gym.Env = None,
                 bandit: EpsilonDeepQGreedy = None,
                 q_function: nn.Module = None, 
                 terminated: Optional[bool] = False
                 ) -> None:

        self.state = state
        self.action = action
         
        self.parent = parent
        self.children = []
            
        self.reward = reward
        self.environment = environment
        self.q_function = q_function
        self.bandit = bandit
        self.terminated = terminated
        
        self.action_space = set(range(self.environment.action_space.n))
    
    @property
    def is_fully_expanded(self) -> bool:
        """ Return true if and only if all child actions have been expanded aka |A| = len(children) """
        return bool(len(self.action_space) == self.children)
    
    @property
    def is_terminal(self) -> bool: 
        return self.terminated
        
    def select(self) -> DeepQNode: 
        """ Recursively looks for node not fully explored and returns it """
        
        if not self.is_fully_expanded or self.is_terminal: # if not fully expand, return this to expand 
            return self 
        
        else: # else, keep recursively looking
            action = self.bandit.select(self.state, self.q_function) # eps greedy with q for action selection
            self.get_outcome_child(action).select()
            
    def get_outcome_child(self, action) -> DeepQNode: 
        """ Selects the next child node after executing an action """
        next_state, reward, terminated, truncated, info  = self.environment.step(action)
        
        for child in self.children: 
            if next_state == child.state: 
                return child
        
        return DeepQNode(next_state, action, self, self.action_space, reward, self.q_function, terminated)
        
    def expand(self, terminated: bool) -> DeepQNode: 
        """ Expand a node if it is not a terminal node """
        if terminated: 
            rand_action = self.environment.action_space.sample()
            return self.get_outcome_child(rand_action)
        
        return self

