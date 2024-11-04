""" Reasoner that uses Two agents: world model agent and action agent to reason through a problem """

from __future__ import annotations
from typing import Callable

from agents.generators import BaseLLMGenerator
from agents.prompts import BasePromptTemplate
from agents.gsm8k.utils import filter_output_type

class LLMWorldModel: 
    def __init__(self, 
                 generator: BaseLLMGenerator, 
                 prompt: BasePromptTemplate, 
                 llm_output_filter: Callable = filter_output_type
                 ) -> None:
        
        self.generator = generator
        self.prompt = prompt
        self.llm_output_filter = llm_output_filter
        
    def init_state(self): 
        ...
        
    def is_terminal(self): 
        ...
        
    def _generate_reward(self): 
        """ Generates the reward for an action """
        
    def _generate_next_state(self): 
        """ Generates the next state in the environment for an action """
        
    def step(self): 
        """ Generates the next state given the current action or state """
        ...
        
    def reset(self): 
        """ Resets the environment to its original state"""
        ...
    
class Actor: 
    ...

class WorldReasoner: 
    
    def __init__(self) -> None:
        pass