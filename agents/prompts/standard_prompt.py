"""
Class for constructing in Llama prompts for LLama-3 Vllm Chat Models 
"""
from __future__ import annotations
import json
import re
import os
import random
from typing import TypedDict, TypeVar, Optional, Any, Callable, Literal, Tuple
from textwrap import dedent
import itertools
import pprint as pp
from enum import Enum
import copy

from agents.utils import BaseConfig
from agents.prompts.base_prompt_template import BasePromptTemplate
from agents import prompt_registry

PROMPT_NAME = 'vllm'
Example = TypeVar('Example')

from typing import List, Dict, Literal
import copy
import pprint as pp

class StandardGSMPromptTemplate(BasePromptTemplate):
    """Question answer prompt template."""
    
    # Base template for system message
    template = """
    You are an intelligent AI that is good at answering math questions. Answer the following math questions given to you.
    As a rule, you must return your answer in the format: #### <your_answer>
    """

    def __init__(self) -> None:
        """Initialize the QuestionAnswerPromptTemplate."""
        
        self._fsl_prompt_base = [{'role': 'system', 'content': self.template.strip()}]  # Initialize base with system prompt
        self.fsl_prompt = copy.deepcopy(self._fsl_prompt_base)

    def add(self, role: Literal['user', 'assistant', 'system'], content: str) -> None: 
        """Add new content to the prompt."""
        self.fsl_prompt.append({'role': role, 'content': content})
        
    def pop(self) -> None: 
        """Pops the most recent interaction from the prompt."""
        self.fsl_prompt.pop(-1)
        
    def reset(self) -> None: 
        """Resets the whole prompt to the base FSL prompt."""
        self.fsl_prompt = copy.deepcopy(self._fsl_prompt_base)

    def preprocess(self, **kwargs) -> str:
        """
        Preprocess the sequence of chats and return a formatted string 
        with roles and contents.

        Returns
        -------
        str
            The formatted prompt as a single string.
        """
        # # Concatenate each role and content
        # formatted_prompt = "\n\n".join(f"{msg['role']}:\n{msg['content']}" 
        #                                for msg in self.fsl_prompt if 'content' in msg and 'role' in msg)
        return self.fsl_prompt

    def __repr__(self) -> str:
        return pp.pformat(self.fsl_prompt)
    
    def __str__(self) -> str:
        return pp.pformat(self.fsl_prompt)
