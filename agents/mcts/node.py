""" BigTree Version of MCTS Node """

from __future__ import annotations
import copy
from typing import Generic, Optional, Callable, Union, TypeVar
import itertools
from rich.console import Console
from rich.table import Table
from io import StringIO
import numpy as np
from bigtree.node.node import Node

# types
Computable = Union[np.ndarray, int, float]
State = TypeVar('State')
Action = TypeVar('Action')
Reward = TypeVar('Reward', bound=Computable)

class MCTSNode(Node, Generic[State, Action]):
    
    id_iter = itertools.count() # iterator; each next returns next step as int

    @classmethod
    def reset_id(cls):
        cls.id_iter = itertools.count() 

    def __init__(self, 
                 state: State = None, 
                 action: Action = None, 
                 reward: Reward = None,
                 parent: "Optional[MCTSNode]" = None,
                 is_terminal: bool = False, 
                 calc_q: Callable[[list[float]], float] = np.mean
                 ) -> None:
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
        # class-level attr
        # tracks how many instances of MCTSNode have been created as a way to show id of node 
        self.id = next(MCTSNode.id_iter)
        self._name = f'{self.id}'
        super().__init__(name=self._name, parent=parent)
        
        # object init attrs
        self._state: State = state
        """ Stores the history state of a prompt """
        self.action: Action = action
        """ Store the action that led to this state as a string """
        self.reward: Reward = reward
        """ Stores the float reward from the action that led to this node state; just one value """
        self.is_terminal = is_terminal
        self.calc_q = calc_q
        # internal attr tracking 
        self._cum_rewards: list[float] = [] # cumulative rewards from rollout
        self._fast_heuristic = 0.0
        
    @property 
    def state(self) -> State: 
        """ Returns deepcopy of state """
        return copy.deepcopy(self._state)
            
    @property 
    def cum_rewards(self) -> list[float | int]: 
        """ Stores the values of the cumulative rewards from each rollout """
        return self._cum_rewards

    # noinspection PyPep8Naming
    @property
    def Q(self) -> float | np.ndarray:
        """ 
        My Q function: 
        Note - if self.cum_rewards = [] aka it has not been visited before, 
                make Q value --> inf aka SUPER exploration (each child node visited at least once)
        
        Default Q calculation is mean of cumulative rewards
        """
        if self.cum_rewards: # if node been explored before 
            return self.calc_q(self.cum_rewards)
        else: # else, Q value to infinity --> explore
            return 0.0
        
    @property
    def depth(self) -> int: 
        """ Computes the depth of the node in the tree """
        depth, node = 0, self
        while self.parent: 
            depth +=1 
            node = node.parent
        return depth
    
    def uct(self, w_exp: float) -> float:
        """ 
        Gets the current UCT value for the node 

        :param node: 

         - Note: cum_rewards = num full rounds (expansion -> simulation -> backprop) involving that node

         - N = Calculates number times parent node visited via cum_rewards

         - n_i = number times child node visited via cum_rewards
        """
        N = max(1, len(self.parent.cum_rewards))  # num times parent node visited -
        n_i = max(1, len(self.cum_rewards))  # num times child node visited
        term = w_exp * np.sqrt(np.log(N) / n_i)  # left term in UCT
        return self.Q + term
    
    def uct_select(self, w_exp: float) -> MCTSNode:
        """ 
        Supposing the node is fully expanded (aka max children), selects and returns the best child node (maxes UCT) out of the children 

        :node: the current node you are at in your tree search

        Note - This is called recursively in "_select" as you traverse the tree
        Note - no fast reward, so node must be fully expanded
        """
        return max(self.children, key=lambda child: child.uct(w_exp))
    
    def terminal_depth_limit(self, depth_limit: int) -> bool:
        """ True if node is terminal or depth limit exceeded """
        return bool(self.is_terminal or self.depth >= depth_limit)
        
    def __str__(self) -> str:
        # Using rich to capture formatted string for __str__
        console = Console(file=StringIO(), width=60)
        table = Table(title=f"NodeID: {self.id}", show_header=True, header_style="bold cyan")
        table.add_column("Attribute", style="dim")
        table.add_column("Value")
        
        table.add_row("State", str(self.state.history))
        table.add_row("Action", str(self.action))
        table.add_row("Parent ID", str(self.parent.id if self.parent else "None"))
        table.add_row("Q-Value", f"{self.Q:.2f}")
        # table.add_row("Reward", f"{self.Q:.2f}")
        table.add_row("Number Children", f"{len(self.children)}")
        console.print(table)
        return console.file.getvalue()
    
    def __repr__(self) -> str:
        # Create a StringIO buffer to capture Rich output
        buffer = StringIO()
        console = Console(file=buffer, width=80, force_terminal=True)
        # Build a Rich Table or any other Rich component
        table = Table(title=f"Node ID: {self.id}", show_header=False, box=None)
        # Add rows to the table
        table.add_row("[bold]Parent ID:[/]", str(self.parent.id if self.parent else "None"))
        table.add_row("[bold]Action:[/]", str(self.action))
        table.add_row("[bold]Q-Value:[/]", f"{self.Q:.2f}")
        table.add_row("[bold]Reward:[/]", str(self.reward))
        table.add_row("[bold]Is Terminal:[/]", str(self.is_terminal))
        table.add_row("[bold]Depth:[/]", str(self.depth))
        table.add_row("[bold]Num Children:[/]", str(len(self.children)))
        # If state is complex, you might summarize it
        state_length = len(self.state.history) if self.state and hasattr(self.state, 'history') else "None"
        table.add_row("[bold]State Length:[/]", str(state_length))
        # Render the table to the console (which writes to the buffer)
        console.print(table)
        # Get the string from the buffer
        rich_output = buffer.getvalue()
        # Return the string
        return rich_output
    
NodePath = list[MCTSNode]
""" List of nodes representing a path in MCTS Tree """