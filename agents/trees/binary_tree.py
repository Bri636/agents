from __future__ import annotations

'''Binary Tree'''

from typing import Any

class Node: 
    def __init__(self, data: Any) -> None:
        
        self.data = data
        self.left = None
        self.right = None
        
        
        
class BinaryTree: 
    '''Simple General Binary Tree'''
    
    def __init__(self) -> None:
        self.root = None
        
    def insert(self, new_data: Any) -> None: 
        '''Inserts new data as a node'''
        
        if self.root: 
            self._insert_recursive(self.root, Node(new_data))
        else: 
            self.root = Node(new_data) # set root node if none
            
    def _insert_recursive(self, current_node: Node, new_data: Any) -> None: 
        '''Recursively finds the correct place for insertion'''
        
        if current_node.left is None: 
            current_node.left = Node(new_data)
        elif current_node.right is None: 
            current_node.right = Node(new_data)
        else: 
            self._insert_recursive(current_node.left, new_data)
            
    def inorder_traversal(self, node: Node) -> Node: 
        '''Traverse the tree in order'''
        if node: 
            self.inorder_traversal()
            
            
            
            
            
            
################# new tree 

import math
import random

class Node:
    def __init__(self, state, parent=None):
        self.state = state  # Current game state at this node
        self.parent = parent  # Parent node
        self.children = []  # List of child nodes
        self.visits = 0  # Number of times this node has been visited
        self.value = 0  # Total value accumulated at this node

    def is_fully_expanded(self):
        """Check if all possible actions from this state have been expanded."""
        return len(self.children) == len(self.state.get_possible_moves())

    def best_child(self, exploration_weight=1.0):
        """Use the UCB1 formula to select the best child."""
        choices_weights = [
            (child.value / child.visits) + exploration_weight * math.sqrt(math.log(self.visits) / child.visits)
            for child in self.children
        ]
        return self.children[choices_weights.index(max(choices_weights))]

    def expand(self):
        """Expand a new child node for an untried action."""
        possible_moves = self.state.get_possible_moves()
        for move in possible_moves:
            if move not in [child.state for child in self.children]:
                new_state = self.state.make_move(move)
                new_node = Node(new_state, parent=self)
                self.children.append(new_node)
                return new_node
        raise Exception("No moves left to expand")

    def backpropagate(self, result):
        """Backpropagate the result from simulation to update the node statistics."""
        self.visits += 1
        self.value += result
        if self.parent:
            self.parent.backpropagate(result)

class MCTS:
    def __init__(self, exploration_weight=1.0):
        self.exploration_weight = exploration_weight

    def search(self, initial_state, num_simulations=1000):
        """Perform MCTS starting from the initial state."""
        root = Node(initial_state)

        for _ in range(num_simulations):
            node = self._select(root)
            if not node.state.is_terminal():
                node = node.expand()
            result = self._simulate(node.state)
            node.backpropagate(result)

        # Return the best action (child of root with most visits)
        return max(root.children, key=lambda n: n.visits).state

    def _select(self, node):
        """Select a node to expand."""
        while not node.state.is_terminal() and node.is_fully_expanded():
            node = node.best_child(self.exploration_weight)
            
        return node

    def _simulate(self, state):
        """Simulate the game by randomly playing until the game ends."""
        
        current_state = state
        while not current_state.is_terminal():
            move = random.choice(current_state.get_possible_moves())
            current_state = current_state.make_move(move)
            
        return current_state.get_result()

# Game State Example (You need to implement this for your specific game)
class GameState:
    def get_possible_moves(self):
        """Return all possible moves from this state."""
        pass

    def make_move(self, move):
        """Return the new state after making a move."""
        pass

    def is_terminal(self):
        """Check if the game has ended."""
        pass

    def get_result(self):
        """Return the result of the game (1 for win, -1 for loss, 0 for draw)."""
        pass

# Example usage:
# initial_state = YourGameState()  # Initialize your game's state
# mcts = MCTS(exploration_weight=1.4)
# best_next_state = mcts.search(initial_state, num_simulations=1000)

        
        
            

            
        
        
        