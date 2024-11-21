from __future__ import annotations

""" BigTree Version of MCTS Node """
import pickle
from os import PathLike
import pickle
import math
import copy
from typing import Generic, Optional, NamedTuple, Callable, Hashable, Any, Literal, Tuple, NewType, Union, TypeVar
import itertools
from abc import ABC
from collections import defaultdict
from rich.console import Console
from rich.table import Table
from io import StringIO
import numpy as np
from tqdm import trange
import gymnasium as gym
import random
from textwrap import dedent
from rich.tree import Tree

from agents.mcts.base import (SearchAlgorithm, WorldModel, SearchConfig, 
                                                State, Action, Example, Trace)
from agents.utils import calculate_returns
from agents.prompts.base_prompt_template import BasePromptTemplate
from agents.prompts.llama_prompt import GSMLlamaPromptTemplate
import gymnasium as gym 
import ale_py
from rich.pretty import pprint as rpprint
import pprint as pp

from bigtree.node.node import Node

LLMNodeState = Union[BasePromptTemplate, GSMLlamaPromptTemplate]
""" BasePromptTemplate or GSMLLamaPromptTemplate"""

LLMNodeAction = str
""" String that is an action of the LLM """

LLMNodeReward = Union[float, np.ndarray, int]
""" Reward value from an action """

class BTMCTSNode(Node, Generic[State, Action, Example]):
    
    id_iter = itertools.count() # iterator; each next returns next step as int

    @classmethod
    def reset_id(cls):
        cls.id_iter = itertools.count() 

    def __init__(self, 
                 state: LLMNodeState = None, 
                 action: LLMNodeAction = None, 
                 reward: LLMNodeReward = None,
                 parent: "Optional[BTMCTSNode]" = None,
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
        self.id = next(BTMCTSNode.id_iter)
        self._name = f'{self.id}'
        super().__init__(name=self._name, parent=parent)
        
        # object init attrs
        self._state: LLMNodeState = state
        """ Stores the history state of a prompt """
        self.action: LLMNodeAction = action
        """ Store the action that led to this state as a string """
        self.reward: LLMNodeReward = reward
        """ Stores the float reward from the action that led to this node state; just one value """
        self.is_terminal = is_terminal
        self.calc_q = calc_q
        
        # internal attr tracking 
        self._cum_rewards: list[float] = [] # cumulative rewards from rollout
        # heurestic reward 
        self._fast_heuristic = 0.0
        
    @property 
    def state(self) -> LLMNodeState: 
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