"""Interface for all agents to follow."""

from __future__ import annotations

from typing import Protocol, Callable, Optional, Any, TypeVar, Type, Union, runtime_checkable
from abc import ABC, abstractmethod
from functools import singledispatch

from agents.utils import BaseConfig
from agents.generators.base_generator import BaseLLMGenerator
from agents.prompts.base_prompt_template import BasePromptTemplate

T = TypeVar('T')

class BaseGenerativeAgent(Protocol): 
    """
    Summary: 
    ========
    Base Container Class for Generative Agents - This Agent Is Used for Inference Only:
    
    Base Attributes: 
    =============== 
    - config: Any configurations for the Agent
    - generator: LLM generator class used for inference 
    - prompt_template: str template that can batch preprocess dictionary kwargs with .format()
    - parser: json parser that takes out 
    """
    
    def __init__(self, 
                 config: BaseConfig, 
                 generator: BaseLLMGenerator,
                 prompt_template: BasePromptTemplate,
                 parser: Optional[Union[Type, Callable]], 
                 **kwargs
                 ) -> None:
        """Initialize the generator with the configuration."""
        
    def batch_generate(self, **kwargs) -> list[str]: 
        """ Some kind of generation function that returns a batch of responses. """
    