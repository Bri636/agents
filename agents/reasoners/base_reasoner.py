""" Base Class for Reasoner """

from __future__ import annotations
from abc import ABC, abstractmethod
from typing import Callable, Self
from agents.reasoners import T
from agents.generators.base_generator import BaseLLMGenerator

class BaseReasoner: 
    
    registry = {}
    
    def __init__(self) -> None:
        pass
    
    @abstractmethod
    def generate_answer(self): 
        """ Generates Raw Text Answer Given a Question """
        pass
    
    @abstractmethod
    def reset_pass(self): 
        """ Resets the prompts for a reasoner """
        pass
        
    @classmethod
    def register(cls: Self, name: str = None) -> Callable: 
        """ Registers a reasoner class in the base reasoners registry """
        def decorator(subclass: T) -> T: 
            cls.registry[name or subclass.__name__] = subclass
            return subclass
        return decorator
    
    @classmethod
    def initialize(cls: Self, generator: BaseLLMGenerator, llm_output_filter: Callable, **kwargs) -> Self: 
        """ Initializes a Reasoner from at least a generator and an llm_output_filter callable """
        pass
    
    def batch_generate_answer(self): 
        """ Batch generated answers """
        pass