""" Base Class for Reasoner """

from __future__ import annotations
from abc import ABC, abstractmethod
from typing import Callable, Self, Tuple

from agents.reasoners.types import T
from agents.generators.base_generator import BaseLLMGenerator
from agents.utils import ConfigLike

class BaseReasoner: 
    
    _registry = {}
    
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
            cls._registry[name or subclass.__name__] = subclass
            return subclass
        return decorator
    
    def batch_generate_answer(self): 
        """ Batch generated answers """
        pass
    
    def __init_subclass__(cls, *, name: str):
        super().__init_subclass__()
        # if not in registry, add to it 
        if not cls._registry.get(name):
            cls._registry[name] = (cls)
            
    @classmethod
    def get_registery(cls: Self) -> dict[str, Self]: 
        return cls._registry