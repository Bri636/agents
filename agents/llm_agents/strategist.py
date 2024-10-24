'''Chain for action inference'''

from __future__ import annotations
from textwrap import dedent
from pydantic import BaseModel, Field
import torch
import numpy as np
import pprint as pp
import json
import lightning as pl

from typing import Protocol, Callable, Optional, Any, TypeVar, Union, runtime_checkable, Tuple
from abc import ABC, abstractmethod
from copy import deepcopy
from enum import Enum
from pydantic import BaseModel, computed_field, Field
from dataclasses import dataclass, field

from agents.utils import BaseConfig
from agents.generators.base_generator import BaseLLMGenerator
from agents.prompts.base_prompt import BasePromptTemplate
from agents.parsers import LLMOutputParser

from agents.prompts.strategy_prompt import (StrategyInputPayload,
                                            StrategyOutputPayload,
                                            StrategyPromptTemplate)
from agents.llm_agents.agent_utils import LactChainAgentMessage
from agents import agent_registry

T = TypeVar('T')

AGENT_NAME = 'Strategist'


class GenerationConfig(BaseConfig):
    ''' Settings for how agent should generate output '''
    verbose: bool = False
    num_tries: int = 5


@dataclass  # I use dataclass since pydantic seems to cause error for non-basemodel classes
class LactChainStrategyAgentContainer:
    ''' Config for running the python agent '''
    _name: str = AGENT_NAME
    generation_config: GenerationConfig = field(
        default=None,
        metadata='Config for how agent should generate code'
    )
    action_agent_cls: LactChainStrategyChain = field(
        default=None,
        metadata=' Action Agent cls '
    )
    prompt_template_cls: StrategyPromptTemplate = field(
        default=None,
        metadata='prompt template'
    )
    llm_output_parser: Optional[Union[LLMOutputParser, Callable]] = field(
        default=None,
        metadata='optional: parser that parses action agent output into objects like a dict'
    )
    solver: Optional[Union[Callable]] = field(
        default=None,
        metadata='optional: solver that maps parsed outputs into a predefined set of actions such as in alpha-proof'
    )
    input_payload_cls: StrategyInputPayload = field(
        default=None,
        metadata='Payload container that forms inputs for the agent'
    )
    output_payload_cls: StrategyOutputPayload = field(
        default=None,
        metadata='Payload container that forms outputs from the agent'
    )
    message_cls: LactChainAgentMessage = field(
        default=None,
        metadata='Class for formatting agent messages'
    )

    def __post_init__(self) -> None:
        pass


@agent_registry.register(AGENT_NAME,
                         cls_container=LactChainStrategyAgentContainer,
                         cls_payload={
                             'generation_config': GenerationConfig,
                             'message_cls': LactChainAgentMessage
                         })
class LactChainStrategyChain:
    ''' Strategy Chain Class '''

    def __init__(self,
                 generation_config: GenerationConfig,
                 generator: BaseLLMGenerator,
                 prompt_template_cls: StrategyPromptTemplate,
                 output_payload_cls: StrategyOutputPayload,
                 message_cls: Optional[LactChainAgentMessage],
                 llm_output_parser: Optional[Union[LLMOutputParser,
                                                   Callable]] = None,
                 solver: Optional[Union[Callable]] = None,
                 **kwargs
                 ) -> None:

        self._generator = generator
        self._prompt_template = prompt_template_cls()

        self.llm_output_parser = llm_output_parser
        self.solver = solver
        self.output_payload_cls = output_payload_cls
        self.message_cls = message_cls

        generation_config = generation_config()
        self.verbose = generation_config.verbose
        self.num_tries = generation_config.num_tries

    @property
    def generator(self):
        return self._generator

    @generator.setter
    def generator(self, new_generator: str) -> None:
        '''Swap backends then update rest of agent'''

        print(
            f'Swapping generator mode from {self.generator} ===> {new_generator}')
        self._generator = new_generator  # property that allows us to swap out backend

    @property
    def prompt_template(self):
        return self._prompt_template

    @prompt_template.setter
    def prompt_template(self, new_prompt_template: StrategyPromptTemplate) -> None:
        self._prompt_template = new_prompt_template

    def batch_generate(self, payloads: StrategyInputPayload | list[StrategyInputPayload]) -> list[str]:
        ''' Generates a batch of raw llm_outputs, their organized output payloads, and bools 
        for if they were successfully parsed or not.  '''
        
        if isinstance(payloads, StrategyInputPayload):
            payloads = [payloads]

        prompts = [self.prompt_template.preprocess(**payload.model_dump())
                   for payload in payloads]

        llm_outputs: list[str] = self.generator.generate(prompts)

        # # Parse and create output payloads in one pass
        # parsed_results: list[Tuple[bool, StrategyOutputPayload]] = [
        #     (success, self.output_payload_cls(**parsed))
        #     for success, parsed in map(self.llm_output_parser, llm_outputs)
        # ]

        # sucesses, out_payloads = zip(*parsed_results)
        
        return llm_outputs 
