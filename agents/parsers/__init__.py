from __future__ import annotations

''' Importing parsers '''
from typing import Callable

from agents.parsers.parsers import load_json, strip_json

PARSING_STRATEGIES = {
    'dict': load_json, 
    'json': strip_json
}



def get_parser() -> Callable: 
    ...