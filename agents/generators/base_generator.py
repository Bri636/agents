"""Interface for all language model generators to follow."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Self, Tuple, Union
from pydantic import BaseModel
from abc import ABC, abstractmethod

from agents.utils import BaseConfig
from agents.utils import ConfigLike
from agents.generators.chat_prompt import ChatMessageSequence

@dataclass
class GeneratorOutput:
    """ 
    Output of a Generator 
    
    Attributes:
    ========== 
        - success (bool)
        - output (str)
        - messages (ChatMessageSequence)
    """
    success: bool 
    output: str
    messages: ChatMessageSequence

class BaseLLMGenerator(ABC):
    """Generator protocol for all generators to follow."""
    
    _registry: dict[str, Tuple[BaseLLMGenerator, ConfigLike]] = {}
        
    @abstractmethod
    def generate(self, prompts):
        """Generate response text from prompts.

        list[str]
            A list of responses generated from the prompts
            (one response per prompt).
        """
        ...
        
    @abstractmethod
    def batch_generate(self, prompts): 
        ...
        
    def __init_subclass__(cls, *, name: str, config: ConfigLike = None, **kwargs):
        super().__init_subclass__(**kwargs)
        # add new class if not in registry 
        assert config, f''' LLM Generator subclasses must have a config associated with it '''
        # if not in registry, add to it 
        if not cls._registry.get(name):
            cls._registry[name] = (cls, config)
            
    @classmethod
    def get_registery(cls) -> dict[str, Tuple[BaseLLMGenerator, ConfigLike]]: 
        return cls._registry
    
            