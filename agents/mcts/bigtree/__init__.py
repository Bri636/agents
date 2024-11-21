""" Type initialization and imports for bigtree alg. """

from __future__ import annotations

import numpy as np
from typing import TypeVar, Union, Callable

from agents.prompts.base_prompt_template import BasePromptTemplate
from agents.prompts.llama_prompt import GSMLlamaPromptTemplate

Prompt = Union[BasePromptTemplate, GSMLlamaPromptTemplate]
""" Prompt for an agent """

State = Union[BasePromptTemplate, GSMLlamaPromptTemplate]
""" State of a node Also corresponds to Prompt in this case """

Action = str
""" Action of the LLM, most likely an str """

Question = str 
""" Action """

Computable = Union[list[float], float, int, np.ndarray]
""" Values that are computable via function such as np.mean() """

T = TypeVar('T')


# def build_bigtree_mcts()