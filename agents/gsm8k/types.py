""" Types Related to GSM8K """

from __future__ import annotations

from typing import TypeVar, TypedDict

class GSM8KProblem(TypedDict):
    """ 
    Question Type for GSM8K 
    
    Dict({
        'question': ..., 
        'answer': ...
    })
    
    :param question: str question
    :param answer: str answer
    """
    question: str
    answer: str