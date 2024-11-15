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
from rich.pretty import pprint as rpprint
import pprint as pp


