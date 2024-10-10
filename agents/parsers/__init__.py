from __future__ import annotations

''' Importing parsers '''
from typing import Callable, Any, Literal

from agents.parsers.parsers import LLMOutputParser, load_json, strip_json

STRATEGIES = {
    'dict': load_json,
    'json': strip_json
}

def get_parser(strategy: Literal['dict', 'json']) -> LLMOutputParser:
    ''' Function for initializing parser '''

    parser = STRATEGIES.get(strategy)  # type: ignore[arg-type]
    if not parser:
        raise ValueError(
            f'Unknown parsing strategy: {strategy}.'
            f' Available: {set(STRATEGIES.keys())}',
        )
        
    return LLMOutputParser(parser)