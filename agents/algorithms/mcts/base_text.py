from __future__ import annotations

'''Basic Monte Carlo Tree Search using BigTree Package'''

from typing import Any
from bigtree import Node, BaseNode
import string
import random

from abc import ABC, abstractmethod

class BaseNode(ABC):
    '''Container node class for MCTS that wraps around BaseNode'''
    
    def __init__(self) -> None:
        super().__init__()
        '''Attributes goes here'''
        
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
    
    @abstractmethod
    def expand(self) -> None: 
        '''Expand a new child node and add it to children'''
        pass
    
    @abstractmethod
    def backpropagate(self) -> None: 
        '''Update all of the ancestors up until root node'''
        pass
        
        
class BaseMCTS(ABC): 
    '''Container for executing MCTS'''
    
    def __init__(self, exploration_weight: float) -> None:
        super().__init__()
        self.exploration_weight = exploration_weight
        
    @abstractmethod
    def search(self) -> Node: 
        '''Iteratively expand node until full, then simulates until end. Then, backprop.'''
        pass
    
    @abstractmethod
    def _select(self) -> Node: 
        '''Selects node to expand based on criteria such as UCB or Q-value'''
        pass
    
    @abstractmethod
    def _simulate(self) -> Node: 
        '''Rollout policy implementation'''
        pass



class PlanningTree: 
    '''Stores actions and errors'''
    
    def __init__(self, root: Node) -> None:
        
        self.root = root
        self.node_names: list[str] = list(string.ascii_lowercase) # stores the alphabet as a list for marking child nodes
        
        
    def add_children(self, node: Node, children: list[Any] | dict[str, Any]) -> None: 
        '''Add an iteratable of children nodes to a specific node'''
        
        self.children = []
        
        for idx, child in enumerate(children): 
            
            if isinstance(children, list): 
                node = Node(self.node_names[idx], child)
            elif isinstance(children, dict): 
                data = {child, children[data]}
                node = Node(self.node_names[idx], data)
                
            self.children.append(node)
        
        node.children = self.children
        
        
    def traverse_to_leaf(self) -> None: 
        '''Go to specific leaf node'''
        
        
        ...
    
