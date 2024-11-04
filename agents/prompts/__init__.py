from __future__ import annotations

''' Factory functions for creating prompts, parsers, and output classes '''

from typing import Union
from agents import prompt_registry
from agents.registry import import_submodules
from agents.extra.prompts.action_prompt_template import Instructions
from agents.prompts.base_prompt_template import BasePromptTemplate

import_submodules(__name__) # trigger import of submodules of this module so we auto-register classes

# ActionAgentPrompts = Union[
#     tuple(elem['class'] for elem in prompt_registry._registry.values())
# ]

def get_instruction(name: str = "COT") -> str: 
    ''' Retrieves an instruction from set of instructions for actor agent '''
    
    supported = list(Instructions.__members__.keys())
    if name not in supported: 
        raise ValueError(f'Unsupported instruction {name}. Choose from {supported}')
    
    instruction = Instructions[name].value
    return instruction