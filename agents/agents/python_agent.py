"""Interface for all agents to follow."""

from __future__ import annotations

from typing import Protocol, Callable, Optional, Any, TypeVar, Type, Union, runtime_checkable
from abc import ABC, abstractmethod

from agents.configs import BaseConfig
from agents.base_classes import BaseLLMGenerator
from agents.base_classes import BasePromptTemplate
from agents.base_action_agent import BaseActionAgent

T = TypeVar('T')

class PythonCodeAgent(BaseActionAgent): 
    
    def __init__(self, 
                config: BaseConfig, 
                generator: BaseLLMGenerator,
                prompt_template: BasePromptTemplate,
                parser: Optional[Union[Type, Callable]], 
                solver: Optional[Union[Type, Callable]], 
                **kwargs
                ) -> None:
        
        ...
        
    def preprocess(self, **kwargs) -> str: 
        """Preprocesses raw input with prompt_template and returns a string"""
        
        pass
        
    def parse_outputs(self, **kwargs) -> list[Any]: 
        """ Loops through raw outputs and parses it with the parser """
        pass
    def map_actions(self, **kwargs) -> list[Any]: 
        """ Optional that maps parsed outputs to actions that can be executed in the environment """
        pass
    
    def iteratively_generate(self, **kwargs) -> None: 
        """ Execute agents projects """
        pass
    
    def execute(self, **kwargs) -> None: 
        """ Execute an action """
        pass
    

    