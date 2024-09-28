from __future__ import annotations
from typing import Any, TypeVar

'''MCTS Tree implementation'''

import random

from bigtree import Node
from bigtree.utils import exceptions
import gymnasium as gym

from agents.algorithms.mcts.base_text import BaseNode


MCTSNodeLike = TypeVar("MCTSNodeLike", bound="MCTSNode")

class MCTSNode:
    '''Overwritten Node from: https://github.com/kayjan/bigtree/blob/master/bigtree/node/node.py'''

    def __init__(self,
                 state: Any,
                 action: Any,
                 parent: MCTSNodeLike = None,
                 children: list[MCTSNodeLike] = [],
                 action_space: set[Any] = ()
                 ) -> None:
        super().__init__()

        self.state = state  # current state of node
        self.action = action  # action executedto generate node

        self._parent: MCTSNodeLike = parent
        self._children: list[MCTSNodeLike] = list(children)
        
        if isinstance(action_space, gym.spaces.discrete.Discrete): 
            self.action_space: set[Any] = set(range(action_space.n))
        else: 
            self.action_space: set[Any] = action_space

        # computed from some criteria measure
        self.value = None
        self.number_visits = 0
        self._executed_actions: list[Any] = []

    @property
    def parent(self) -> MCTSNodeLike:
        return self._parent

    @property
    def children(self) -> MCTSNodeLike:
        return self._children

    @property
    def is_fully_expanded(self) -> bool:
        '''Tests if node is fully expanded by seeing if number children is same as number of actions'''
        return bool(len(self.action_space) == self.children)

    @property
    def best_child(self, exploration_weight: float) -> MCTSNodeLike:
        '''Selects the best child node; use after node is fully expanded'''
        rand = random.random()
        if rand > exploration_weight:  # explore
            return random.sample(self.children, 1)[0]  # select node randomly
        else:
            return max([child_node for child_node in self.children],
                       key=lambda x: x.value)

    def add_child(self, child_node: MCTSNodeLike) -> None:
        '''Adds child node'''
        self.children.append(child_node)

    def expand(self, action: Any, environment: gym.Env) -> MCTSNodeLike:
        '''Apply one action in environment, then expand that as child node.'''

        if action in self._executed_actions:
            pass
        else:
            next_state = environment.step(action)
            child_node_data = {'state': next_state,
                               'action': action,
                               'parent': self,
                               'children': [],
                               'action_space': self.action_space}
            child_node = MCTSNode(**child_node_data)
            
            self.add_child(child_node)
            self._executed_actions.append(action)
            
        return child_node

    def backpropagate(self, simulation_result: Any) -> None:

        self.number_visits += 1
        self.value += simulation_result

        if self.parent:
            self.parent.backpropagate(simulation_result)


if __name__ == "__main__": 
    
    import gymnasium as gym
    
    from nes_py.wrappers import JoypadSpace
    import gym_super_mario_bros
    from gym_super_mario_bros.actions import SIMPLE_MOVEMENT
    
    env = gym_super_mario_bros.make('SuperMarioBros-v0')
    env = JoypadSpace(env, SIMPLE_MOVEMENT)
    
    state = env.reset()
    action = env.action_space.sample()
    next_state, reward, done, info = env.step(action)
    
    root_node = MCTSNode(state=next_state, 
                         action=action, 
                         parent=None, 
                         children=[], 
                         action_space=env.action_space)
    
    root_node.add_child()
    
    breakpoint()