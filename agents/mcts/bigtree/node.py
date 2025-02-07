""" Node """

from __future__ import annotations
from typing import Generic, TypeVar, Union, Self, Optional, Callable
from bigtree.node.node import Node
import numpy as np
import itertools
import copy
import math

# types
Computable = Union[np.ndarray, int, float]
State = TypeVar('State')
Action = TypeVar('Action')
Reward = TypeVar('Reward', Computable)

class MCTSNode(Node):
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
    id_iter = itertools.count()

    def __init__(self,
                 state: State,
                 action: Action,
                 reward: Reward,
                 parent: Optional[Self] = None,
                 is_terminal: bool = False,
                 q_function: Callable = np.mean,
                 w: Computable = 0.5,
                 **kwargs):
        
        self.id = next(MCTSNode.id_iter)
        self._name = f'{self.id}'
        super().__init__(name=self._name, parent=parent)
        
        self._state = state, 
        self.action = action
        self.reward = reward
        self.parent = parent
        self.is_terminal = is_terminal
        self.q_function = q_function
        self.w = w
        
        # internal 
        self._cumulative_rewards: list[Computable] = []
        
    @property
    def state(self) -> State: 
        """ Returns a deepcopy of the node state """
        return copy.deepcopy(self._state)
    
    @property
    def cumulative_rewards(self) -> list[Computable]: 
        """ List of cumulative rewards aka stores a list of the summed rewards <=> return """
        return self._cumulative_rewards
    
    @property
    def Q(self) -> Computable: 
        if self._cumulative_rewards: 
            return self.q_function(self.cumulative_rewards)
        else: 
            return 0.0
        
    @property
    def uct(self) -> Computable:
        """ Computes UCT value for this node """ 
        N = len(self.parent.cumulative_rewards) # each time we visit a node, we do 1 -> 4 in MCTS, so its proxy
        n_i = max(1, len(self.cumulative_rewards)) # cant be zero 
        uct = self.Q + self.w * math.sqrt( math.log(N) / n_i)
        return uct
    
    @property
    def best_child(self) -> Self: 
        """ Returns the best child for node based on max uct """
        return max(self.children, self.uct)
    
    @property
    def depth(self) -> int: 
        """ Computes the depth of the node in the tree """
        depth, node = 0, self
        while self.parent: 
            depth +=1 
            node = node.parent
        return depth
    
    def terminal_with_depth_limit(self, depth_limit: int) -> bool:
        """ True if node is terminal or depth limit exceeded """
        return bool(self.is_terminal or self.depth >= depth_limit)
    

NodePath = list[MCTSNode]
""" List of Nodes """
        

    

    
        
        
