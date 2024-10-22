from __future__ import annotations

''' Factory functions for creating prompts, parsers, and output classes '''

from typing import Union
from agents import prompt_registry
from agents.registry import import_submodules

import_submodules(__name__) # trigger import of submodules of this module so we auto-register classes

# ActionAgentPrompts = Union[
#     tuple(elem['class'] for elem in prompt_registry._registry.values())
# ]