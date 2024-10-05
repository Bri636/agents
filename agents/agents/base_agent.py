"""Interface for all agents to follow."""

from __future__ import annotations

from typing import Protocol, Callable, Optional, Any, TypeVar, Type, Union, runtime_checkable
from abc import ABC, abstractmethod

from agents.configs import BaseConfig
from agents.base_classes import BaseLLMGenerator
from agents.base_classes import BasePromptTemplate

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
        
    def preprocess(self, **kwargs) -> str: 
        """Preprocesses raw input with prompt_template and returns a string"""
        
    def parse_outputs(self, **kwargs) -> list[Any]: 
        """ Loops through raw outputs and parses it with the parser """
    