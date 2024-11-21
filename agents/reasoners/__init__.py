""" Reasoners for GSM """

from __future__ import annotations

def register_strategy(strategy_dict, name=None):
    """Decorator to register a method as a search strategy."""
    def decorator(func):
        strategy_name = name or func.__name__
        strategy_dict[strategy_name] = func
        return func
    return decorator

class ReasonerStrategies:
    """ Class interface for interacting with sub_reasoners """ 
    strategies = {}
    
    def __init__(self) -> None:
        pass
    
    @classmethod
    def execute_strategy(self):
        ...