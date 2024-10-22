from __future__ import annotations

''' Factory functions for creating prompts, parsers, and output classes '''

from typing import Union
from agents.lactchain import lactchain_prompt_registry
from agents.registry import import_submodules

import_submodules(__name__) # trigger import of submodules of this module so we auto-register classes

ActionAgentPrompts = Union[
    tuple(elem['class'] for elem in lactchain_prompt_registry._registry.values())
]
# ActionAgentInputPayloads = Union[
#     tuple(elem['class'] for elem in input_payload_registry._registry.values())
# ]
# ActionAgentOutputPayloads = Union[
#     tuple(elem['class'] for elem in output_payload_registry._registry.values())
# ]