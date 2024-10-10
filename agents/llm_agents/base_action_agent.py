"""Interface for all agents to follow."""

from __future__ import annotations

from typing import Protocol, Callable, Optional, Any, TypeVar, Type, Union, runtime_checkable
from abc import ABC, abstractmethod
from functools import singledispatch

from agents.configs import BaseConfig
from agents.generators.base_generator import BaseLLMGenerator
from agents.prompts.base_prompt import BasePromptTemplate


T = TypeVar('T')

class BaseActionAgent(Protocol): 
    """
    Summary: 
    ========
    Base Container Class for Action Agents - This Agent Is Used for Inference Only:
    
    Base Attributes: 
    =============== 
    - config: Any configurations for the Agent
    - generator: LLM generator class used for inference 
    - prompt_template: str template that can batch preprocess dictionary kwargs with .format()
    - parser: json parser that takes out 
    - solver: converts str actions into executable moves 
    """
    
    def __init__(self, 
                 config: BaseConfig, 
                 generator: BaseLLMGenerator,
                 prompt_template: BasePromptTemplate,
                 parser: Optional[Union[Type, Callable]], 
                 output_fn: Optional[Union[Type, Callable]], 
                 solver: Optional[Union[Type, Callable]], 
                 **kwargs
                 ) -> None:
        """Initialize the generator with the configuration."""
    
    def generate(self, **kwargs) -> None: 
        """ Generates one interaction round with a LLM """
        pass
    
    def iteratively_generate(self, **kwargs) -> None: 
        """ Iterative generation with parser and optional solver """
    
    def map_actions(self, **kwargs) -> None: 
        """ Optional that maps parsed outputs to actions that can be executed in the environment """
        pass
    
    def execute(self, **kwargs) -> None: 
        """ Execute an action """
        pass