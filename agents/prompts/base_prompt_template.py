"""Interface for all prompts to follow."""

from __future__ import annotations

import json
from typing import Protocol, Any, Callable, Union, Literal
from abc import ABC, abstractmethod
from functools import singledispatch
from pydantic import BaseModel, Field
from langchain.prompts import PromptTemplate
from rich import print_json

from agents.utils import BaseConfig
from agents import prompt_registry

PROMPT_NAME = 'Base' # flag that denotes the type of agent 

class BaseInputPayload(BaseModel):
    ''' Base Input Payload for LLM. Allows for intake different dicts from different agents. 
    Override this with actual attributes you want to prompt to take. 
    '''
    
    
    def update_error(self, **kwargs) -> None: 
        '''
        Optional function to update in order to add error feedback to input payload iteratively
        '''
        pass
    
class BaseOutputPayload(BaseModel):
    ''' Base class for Parsed Outputs. Add fields to this base output class. 
    By default, error field is included. 
    '''
    error: Union[str, None] = Field(
        description='Error message from a parsing error from your code in a previous round',
        default='None'
    )
    
    @classmethod
    def get_output_format(cls) -> str:
        '''Returns the description of the Payload field class as a str'''
        output_dict = {field: cls.model_fields[field].description 
                       for field in cls.model_fields if field != 'error'}
        return json.dumps(output_dict, indent=4)  # Use json.dumps to format it neatly
    
class BasePromptTemplate(ABC):
    """PromptTemplate ABC protocol for all prompts to follow. Use ABC since we want to be strict 
    with how prompts are handled and inherited, and inherit methods

    Attributes: 
    ==========
    CLASS_TYPE: str
        generic class type name for registry
    template: str 
        string prompt template not filled in yet
    """
    
    @abstractmethod
    def add(self, **kwargs) -> None: 
        """ Adds str info to the prompt in some way """
        
    @abstractmethod
    def pop(self, **kwargs) -> None: 
        """ Pops off the most recently added info from the prompt in some way """
        
    @abstractmethod
    def reset(self, **kwargs) -> None: 
        """ Resets the prompt to its original form in some way """

    @abstractmethod 
    def preprocess(self, **kwargs) -> list[Any]:
        """Preprocess the text into prompts.

        Parameters
        ----------
        text : str
            The text to preprocess.
        contexts : list[list[str]], optional
            The contexts to include for each text, by default None.
        scores : list[list[float]], optional
            The scores for each context, by default None.

        Returns
        -------
        list[str]
            The preprocessed prompts.
        """

    def postprocess(self, responses: list[str]) -> list[str]:
        """Postprocess the responses.

        Parameters
        ----------
        responses : list[str]
            The responses to postprocess.

        Returns
        -------
        list[str]
            The postprocessed responses.
        """
        pass
    
# @prompt_registry.register(name=PROMPT_NAME, payloads={
#     'input': BaseInputPayload, 
#     'output': BaseOutputPayload
# })
# class BasePromptTemplate(ABC):
#     """PromptTemplate ABC protocol for all prompts to follow. Use ABC since we want to be strict 
#     with how prompts are handled and inherited, and inherit methods

#     Attributes: 
#     ==========
#     CLASS_TYPE: str
#         generic class type name for registry
#     template: str 
#         string prompt template not filled in yet
#     """

#     CLASS_TYPE: str = "Prompts"

#     template: str = """ // Your Prompt Template Goes Here // """

#     def __init__(self) -> None:
#         """Initialize the prompt with the configuration."""
        
#         chat_template = PromptTemplate.from_template(self.template)
#         self.chat_template = chat_template
        
#     def preprocess(self, **kwargs: dict[str, Any]) -> list[str]:
#         """Preprocess the text into prompts.

#         Parameters
#         ----------
#         text : str
#             The text to preprocess.
#         contexts : list[list[str]], optional
#             The contexts to include for each text, by default None.
#         scores : list[list[float]], optional
#             The scores for each context, by default None.

#         Returns
#         -------
#         list[str]
#             The preprocessed prompts.
#         """
#         self.chat_template.format(**kwargs)

#     def postprocess(self, responses: list[str]) -> list[str]:
#         """Postprocess the responses.

#         Parameters
#         ----------
#         responses : list[str]
#             The responses to postprocess.

#         Returns
#         -------
#         list[str]
#             The postprocessed responses.
#         """
#         pass

#     def __repr__(self) -> str:
#         return f'Chain Prompt: {self.chat_template}'
    
#     def __str__(self) -> str:
#         return f'Chain Prompt: {self.chat_template}'
