""" Base Class for Reasoner """

from __future__ import annotations

from abc import ABC, abstractmethod

class BaseReasoner: 
    
    def __init__(self) -> None:
        pass
    
    @abstractmethod
    def generate_answer(self): 
        """ Generates Raw Text Answer Given a Question """
        
    @abstractmethod
    def reset_pass(self): 
        """ Resets the prompts for a reasoner """