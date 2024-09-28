from __future__ import annotations

import numpy as np
import random
from agents.algorithms.bandits.base import MultiArmedBandit

import gymnasium as gym
from torch import nn, Tensor

class EpsilonGreedy(MultiArmedBandit):
    
    
    def __init__(self, epsilon=0.1):
        self.epsilon = epsilon

    def reset(self) -> None:
        pass

    def select(self, state, actions, qfunction) -> np.array | float:
        
        # Select a random action with epsilon probability
        if random.random() < self.epsilon:
            return random.choice(actions)
        arg_max_q = qfunction.get_argmax_q(state, actions)
        
        return arg_max_q
    
    
class EpsilonDeepQGreedy(MultiArmedBandit):
    '''Uses Q values for selecting best value'''
    
    def __init__(self, epsilon=0.1, environment: gym.Env=None):
        self.epsilon = epsilon
        self.environment = environment
        

    def reset(self) -> None:
        pass

    def select(self, state, qfunction: nn.Module) -> np.array | float | Tensor:
        
        # Select a random action with epsilon probability
        if random.random() < self.epsilon:
            return self.environment.action_space.sample()
        
        q_values: Tensor = qfunction(state)
        arg_max_q = q_values.argmax(dim=0)
        
        return arg_max_q