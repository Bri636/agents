from __future__ import annotations

''' Importing parsers '''
from typing import Callable, Any

from agents.parsers.parsers import LLMOutputParser, load_json, strip_json
from agents.prompts.base_prompt import BaseOutputPayload

STRATEGIES = {
    'dict': load_json,
    'json': strip_json
}

def get_parser(name: str, output_cls: BaseOutputPayload) -> LLMOutputParser:
    ''' Function for initializing parser '''
    
    parser = STRATEGIES.get(name)  # type: ignore[arg-type]
    if not parser:
        raise ValueError(
            f'Unknown parsing strategy: {name}.'
            f' Available: {set(STRATEGIES.keys())}',
        )
        
    return LLMOutputParser(output_cls, parser)