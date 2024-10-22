from __future__ import annotations

""" This is a dummy file that contains functions that we will toss into our agent to use """

from typing import Callable
from dataclasses import dataclass

class MyClass: 
    ''' Some dataclass for greetings '''
    hello: str = 'Hi'
    
def add_two_nums(x: int, y: int) -> int: 
    """ Function that adds two numbers together """
    return int(x + y)

class Calculator: 
    """ This is a container class that initializes with some calculator function like add_two_nums
    and then can execute it via calculate
    """
    def __init__(self, add_fn: Callable) -> None:
        
        self.add_fn = add_fn
        
    def calculate(self, **kwargs): 
        
        return self.add_fn(**kwargs)
    
    
    