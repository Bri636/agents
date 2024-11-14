from __future__ import annotations

""" Tests with MCTS Node """
import pickle
from os import PathLike
import pickle
import math
from copy import deepcopy
from typing import Generic, Optional, NamedTuple, Callable, Hashable, Any, Literal, Tuple
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

from agents.algorithms.tree_search.base import (SearchAlgorithm, WorldModel, SearchConfig, 
                                                State, Action, Example, Trace)
from agents.utils import calculate_returns
from agents.algorithms.tree_search.mcts_simple import MCTS, MCTSNode
from agents.prompts.llama_prompt import GSMLlamaPromptTemplate
import gymnasium as gym 
import ale_py

class MCTSNode(Generic[State, Action, Example]):
    
    id_iter = itertools.count() # iterator; each next returns next step as int

    @classmethod
    def reset_id(cls):
        cls.id_iter = itertools.count() 

    def __init__(self, 
                 state: Optional[State], 
                 action: Optional[Action], 
                 reward: float = None,
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
        self.id = next(MCTSNode.id_iter)
        
        self.cum_rewards: list[float] = [] # rewards from rollout
        self.is_terminal = is_terminal
        self.action = action
        self.state: GSMLlamaPromptTemplate | Any = state
        self.reward = reward
        self.parent = parent
        self.children: 'Optional[list[MCTSNode]]' = []
        self.calc_q = calc_q
        
        # if no parent => root 
        if parent is None:
            self.depth = 0
        else:
            self.depth = parent.depth + 1 

    # noinspection PyPep8Naming
    @property
    def Q(self) -> float:
        """ 
        My Q function: 
        Note - if self.cum_rewards = [] aka it has not been visited before, 
        make Q value --> inf aka SUPER exploration (each child node visited at least once)
        
        """
        if self.cum_rewards: # if node been explored before 
            return self.calc_q(self.cum_rewards)
        else: # else, Q value to infinity --> explore
            return np.inf
        
    def __str__(self) -> str:
        # Using rich to capture formatted string for __str__
        console = Console(file=StringIO(), width=60)
        table = Table(title=f"Node ID: {self.id}", show_header=True, header_style="bold cyan")
        
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
        # Concise representation for __repr__
        return dedent(f"""
    MCTSNode(id={self.id}, 
    state=dim:{self.state.history}, 
    action={self.action}, 
    Q={self.Q:.2f}, 
    reward={self.reward}, 
    num_children={len(self.children)})
    """)