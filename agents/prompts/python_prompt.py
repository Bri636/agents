from __future__ import annotations

from typing import Any, Union, Optional

""" Basic Python Prompt Implementation """

from agents.configs.configs import BaseConfig
from agents.prompts.base_prompt import BasePromptTemplate, BaseOutputPayload, base_parsing_function

from agents import prompt_registry

class PythonPromptConfig(BaseConfig): 
    _name: str = 'python'
    
@prompt_registry.register(BasePromptTemplate.CLASS_TYPE, config = PythonPromptConfig)
class PythonPrompt(BasePromptTemplate): 
    
    template: str = '''
    You are an intelligent coding agent that is an expert in running python workflows. 
    Your goal is to help me execute python code that I will give you based on a task I specify. 
    
    At each round of conversation, I will provide you with the following: 
    
    Task: 
    // ... //
    
    Code Functions: 
    // Python functions that I currently have, along with their descriptions // 
    
    
    
    
    
    
    '''
    
    def __init__(self, config: BaseConfig) -> None:
        super().__init__(config)
        
        
    def preprocess(
        self,
        text: str | list[str],
        contexts: list[list[str]] | None = None,
        scores: list[list[float]] | None = None
        ) -> list[str]:
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
        ...

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
        ...