""" Reasoners for GSM """

from __future__ import annotations
from typing import TypeVar

T = TypeVar('T')

Chat = list[dict[str, str]]
""" One Single Chat interaction """

def register_strategy(strategy_dict, name=None):
    """Decorator to register a method as a search strategy."""
    def decorator(func):
        strategy_name = name or func.__name__
        strategy_dict[strategy_name] = func
        return func
    return decorator


