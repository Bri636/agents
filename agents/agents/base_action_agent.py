"""Interface for all agents to follow."""

from __future__ import annotations

from typing import Protocol, Callable, Optional, Any, TypeVar, Type, Union, runtime_checkable
from abc import ABC, abstractmethod
from functools import singledispatch

from agents.configs import BaseConfig
from agents.base_classes import BaseLLMGenerator
from agents.base_classes import BasePromptTemplate

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
        
    def preprocess(self, **kwargs) -> str: 
        """Preprocesses raw input with prompt_template and returns a string"""
        
    def parse_outputs(self, **kwargs) -> list[Any]: 
        """ Loops through raw outputs and parses it with the parser """
        
    def map_actions(self, **kwargs) -> list[Any]: 
        """ Optional that maps parsed outputs to actions that can be executed in the environment """
        pass
    
    def execute(self, **kwargs) -> None: 
        """ Execute an action """
        pass
    
    def iteratively_generate(self, **kwargs) -> None: 
        """ Execute agents projects """
        pass