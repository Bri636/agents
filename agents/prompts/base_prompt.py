"""Interface for all prompts to follow."""

from __future__ import annotations

import json
from typing import Protocol, Any, Callable, Union, Literal
from abc import ABC, abstractmethod
from functools import singledispatch
from pydantic import BaseModel, Field
from langchain.prompts import PromptTemplate

from agents.configs import BaseConfig
from agents import input_payload_registry, output_payload_registry, llm_output_parser_registry, prompt_registry
from agents.prompts.utils import PARSING_STRATEGIES

AGENT = 'Base'

input_payload_registry.register(name=AGENT)
class BaseInputPayload(BaseModel):
    ''' Base Input Payload for LLM. Allows for intake different dicts from different agents. 
    Override this with actual attributes you want to prompt to take. 
    '''
    
output_payload_registry.register(name=AGENT)
class BaseOutputPayload(BaseModel):
    ''' Base class for Parsed Outputs. Add fields to this base output class. 
    By default, error field is included. 
    '''
    error: Union[dict[str, str], None] = Field(
        description='error message if parsing error'
    )
    
class LLMOutputParserConfig(BaseConfig): 
    ...

llm_output_parser_registry.register(name=AGENT)
class LLMOutputParser:
    ''' Class that parses json outputs, then filters them through a custom output pydantic class 
    to return only the fields specified in the pydantic class. 
    '''

    def __init__(self, output_cls: BaseOutputPayload, parsing_strategy: Literal['dict', 'json']) -> None:

        self.output_cls = output_cls
        self.parser = PARSING_STRATEGIES.get(parsing_strategy)

    def parse_from_output(self, llm_output: str) -> BaseOutputPayload:
        '''Uses llm_output_parser to parse raw string output and then organize it into a ParsedOutput'''

        try:
            parsed_output: dict[str, Any] = self.parser(llm_output)
            parsed_output.update({'error': None})

            return BaseOutputPayload(**parsed_output)

        except Exception as e:
            parsed_output = {'error': e}
            filled_payload: dict[str, None] = {
                k: None for k, _ in
                list(self.output_cls.model_fields.keys())}
            parsed_output.update(filled_payload)

            return BaseOutputPayload(**parsed_output)
        

class BasePromptTemplateConfig(BaseConfig): 
    ''' Base config and class holder for base prompt template. 
    It will compute an llm parser based on what the strategy, input_payload, and output_payload is
    '''
    _name: str = 'base'
    
    strategy: Literal['dict', 'json'] = Field(
        description='What kind of parsing strategy to use', 
        default='dict'
    )
    
    input_payload: BaseInputPayload = Field(
        description='What type of input payload to use', 
        default_factory=BaseInputPayload
    )
    
    output_payload: BaseOutputPayload = Field(
        description='What type of Output payload to use', 
        default_factory=BaseOutputPayload
    )
    

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

    CLASS_TYPE: str = "Prompts"

    template: str = """ // Your Prompt Template Goes Here // """

    def __init__(self, config: BasePromptTemplateConfig) -> None:
        """Initialize the prompt with the configuration."""
        
        parsing_strategy = PARSING_STRATEGIES.get(config.strategy)
        input_payload = config.input_payload
        output_payload = config.output_payload
        
        self.llm_output_parser = LLMOutputParser()
        
        
        chat_template = PromptTemplate.from_template(self.template)
        

    @abstractmethod
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
        pass

    @abstractmethod
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
