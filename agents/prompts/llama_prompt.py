"""
Class for constructing in Llama prompts for LLama-3 Vllm Chat Models 
"""
from __future__ import annotations
import json
import re
import os
import random
from typing import TypedDict, TypeVar, Optional, Any, Callable, Literal, Tuple, List
from textwrap import dedent
import itertools
import pprint as pp
from enum import Enum
import copy
from dataclasses import dataclass

from agents.utils import BaseConfig
from agents.prompts.base_prompt_template import BasePromptTemplate
from agents.prompts.gsm_llama_prompts import BASE, QUESTION, ANSWER
from agents.gsm8k.utils import read_jsonl

class GSM8kPromptDict(TypedDict):
    """ Stores the Components for the Prompt """
    instruction: str
    interactive_examples: list[str]
    useful_examples: list[str]
    question_prefix: str
    subquestion_prefix: str
    overall_question_prefix: str
    answer_prefix: str
    
@dataclass
class PromptMessage:
    """ 
    PromptMessage container for role and content
    
    fields: 
    ======
    * role - Literal['user', 'assistant', 'system']
    * content - str
    """
    role: Literal['user', 'assistant', 'system']
    content: str

def make_fsl_llama(prompt: GSM8kPromptDict, num_fsl_examples: int, agent_type: Literal['question', 'answer']) -> list[dict[str, str]]:
    """ 
    Takes in a GSM8kPromptDict with a loaded instruction and fsl examples 
    and returns a list of {'user': ..., 'content': ...} messages for llama 
    """
    system_instruction = prompt['instruction']
    formatted_examples = []

    # if question agent, ai answers are the questions
    if agent_type == 'question':
        q_role = 'assistant'
        a_role = 'user'
    else:
        q_role = 'user'
        a_role = 'assistant'

    for idx, example_text in enumerate(prompt['interactive_examples']):
        formatted_example = [{"role": "system", "content": system_instruction}]
        # Extract the user question and answer pairs
        lines = example_text.strip().splitlines()
        # Add the main question to the formatted example
        main_question = next(
            line for line in lines if line.startswith("Question"))
        formatted_example.append(
            {"role": "user", "content": main_question.strip().format(idx=idx + 1)})
        # Process sub-questions and answers
        for line in lines[1:]:
            if line.startswith("Question"):
                formatted_example.append(
                    {"role": q_role, "content": line.strip().format(idx=idx + 1)})
            elif line.startswith("Answer"):
                formatted_example.append(
                    {"role": a_role, "content": line.strip().format(idx=idx + 1)})
        formatted_examples.append(formatted_example)

    # Sample the specified number of formatted examples
    indices = random.sample(range(len(formatted_examples)), num_fsl_examples)
    selected_examples: list[list[dict]] = [
        formatted_examples[i] for i in indices]

    return list(itertools.chain(*selected_examples))

class GSMLlamaPromptTemplate(BasePromptTemplate):
    """Question-Answer Prompt Template for Llama."""

    FSL_PROMPTS = {
        'base': BASE,
        'question': QUESTION,
        'answer': ANSWER
    }

    def __init__(
        self,
        fsl_prompt_type: str = 'question',
        num_fsl_examples: int = 1,
        agent_type: Optional[str] = None
    ) -> None:
        """Initialize the GSMLlamaPromptTemplate."""

        # Validate and set the prompt type
        if fsl_prompt_type not in self.FSL_PROMPTS:
            valid_prompts = ', '.join(self.FSL_PROMPTS.keys())
            raise ValueError(f"Invalid prompt type. Choose from: {valid_prompts}")
        self._fsl_prompt_type = fsl_prompt_type

        # Generate the base prompt using the provided parameters
        fsl_prompt_base_raw = self.FSL_PROMPTS[fsl_prompt_type]
        fsl_prompt_base: List[dict[str, str]] = make_fsl_llama(fsl_prompt_base_raw, num_fsl_examples, agent_type)

        # Initialize prompts and history
        self._base_prompt: List[PromptMessage] = [PromptMessage(**item) for item in fsl_prompt_base]
        self._prompt: List[PromptMessage] = copy.deepcopy(self._base_prompt)
        self._history: List[PromptMessage] = []

        # Store initialization parameters
        self._prompt_kwargs = {
            'fsl_prompt_type': fsl_prompt_type,
            'num_fsl_examples': num_fsl_examples,
            'agent_type': agent_type
        }
        
    @classmethod
    def make_from_prompt(cls: T, prompt: GSMLlamaPromptTemplate) -> T: 
        """ Creates a prompt from the kwargs of another prompt """
        prompt = cls(**prompt.prompt_kwargs)
        return prompt

    @property
    def prompt_kwargs(self) -> dict[str, Any]:
        """Returns the keyword arguments used to initialize the prompt."""
        return self._prompt_kwargs.copy()

    @property
    def prompt_type(self) -> str:
        """Returns the type of the prompt."""
        return self._fsl_prompt_type

    @property
    def history(self) -> List[PromptMessage]:
        """Returns a copy of the change history to prevent in-place modification."""
        return copy.deepcopy(self._history)

    def add(self, role: Literal['user', 'assistant', 'system'], content: str) -> None:
        """Add a new message to the prompt."""
        message = PromptMessage(role=role, content=content)
        self._prompt.append(message)
        self._history.append(message)

    def pop(self, indices: List[int] = [-1]) -> None:
        """Remove messages at specified indices from the prompt and adjust the history."""
        indices_set = {i if i >= 0 else len(self._prompt) + i for i in indices}
        indices_set = {i for i in indices_set if 0 <= i < len(self._prompt)}
        # Remove from prompt
        self._prompt = [msg for idx, msg in enumerate(self._prompt) if idx not in indices_set]
        # Adjust history
        self._history = [msg for msg in self._history if msg in self._prompt]

    def reset(self) -> None:
        """Reset the prompt to the base prompt and clear the history."""
        self._prompt = copy.deepcopy(self._base_prompt)
        self._history.clear()

    def preprocess(self) -> List[dict[str, str]]:
        """
        Preprocess the prompt messages for the model.
        Returns a list of dictionaries with 'role' and 'content'.
        """
        return [{'role': msg.role, 'content': msg.content} for msg in self._prompt]

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(prompt={pp.pformat(self._prompt)})"

    def __str__(self) -> str:
        return pp.pformat(self._prompt)

    def copy_history(self, prompt: 'GSMLlamaPromptTemplate') -> None:
        """
        Copy the history from another GSMLlamaPromptTemplate instance.
        If the prompt types differ, swap 'user' and 'assistant' roles in the copied history.
        """
        if not isinstance(prompt, GSMLlamaPromptTemplate):
            raise TypeError("The provided prompt must be an instance of GSMLlamaPromptTemplate.")

        # Determine if role swapping is needed
        if self._fsl_prompt_type != prompt._fsl_prompt_type:
            swapped_history = [
                PromptMessage(
                    role='assistant' if msg.role == 'user' else 'user' if msg.role == 'assistant' else msg.role,
                    content=msg.content
                )
                for msg in prompt.history
            ]
        else:
            swapped_history = copy.deepcopy(prompt.history)

        # Update the current prompt and history
        self._history.extend(swapped_history)
        self._prompt.extend(swapped_history)
        
        
    def inject_strategy(): 
        ...
