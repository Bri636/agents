from __future__ import annotations

'''Base Class for Multi-Arm Bandits'''

from abc import ABC, abstractmethod

class MultiArmedBandit(ABC):
    '''Base class for bandits'''
    
    @abstractmethod
    def select(self, state, actions, qfunction):
        """ Select an action for this state given from a list given a Q-function """
        pass

    @abstractmethod
    def reset(self):
        """ Reset a multi-armed bandit to its initial configuration """
        self.__init__()
        pass