""" Types Related to GSM8K """

from __future__ import annotations

from typing import TypeVar, TypedDict

class GSM8KProblem(TypedDict):
    """ 
    Dictionary that stores a problem for GSM8K 
    
    :param question: str question
    :param answer: str answer
    """
    question: str
    answer: str
    
T = TypeVar('T')