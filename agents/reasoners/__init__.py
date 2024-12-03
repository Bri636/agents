""" Reasoners for GSM """

from __future__ import annotations
from typing import TypeVar
from agents.reasoners.base_reasoner import BaseReasoner
from agents.reasoners.reasoner import LLMReasoner
from agents.reasoners.wm_reasoner import WorldReasoner
from agents.reasoners.wm_mcts_reasoner import MCTSWorldReasoner

def register_strategy(strategy_dict, name=None):
    """Decorator to register a method as a search strategy."""
    def decorator(func):
        strategy_name = name or func.__name__
        strategy_dict[strategy_name] = func
        return func
    return decorator


