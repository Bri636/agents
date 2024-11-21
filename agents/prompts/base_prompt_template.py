"""Interface for all prompts to follow."""

from __future__ import annotations

import json
from typing import Protocol, Any, Callable, Union, Literal, Self
from abc import ABC, abstractmethod
from functools import singledispatch
from pydantic import BaseModel, Field
from langchain.prompts import PromptTemplate
from rich import print_json

from agents.utils import BaseConfig
    
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
    
    def copy_history(self, prompt: Self) -> None: 
        """
        Sets the current prompt and history based on another BasePromptTemplate instance.
        If the current prompt type is the opposite of the input prompt type, swaps 'user' and 'assistant' roles
        in the copied history and appends the history to the current fsl_prompt.

        Parameters
        ----------
        prompt : BasePromptTemplate
            The instance from which to copy the prompt and history.
        """
    