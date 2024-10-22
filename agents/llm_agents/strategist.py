from __future__ import annotations

'''Chain for action inference'''

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
from agents import agent_registry

T = TypeVar('T')

AGENT_NAME = 'Strategist'

class LactChainAgentMessage(BaseModel):
    ''' Container class for displaying agent messages based on success or failure '''
    success: bool
    num_tries: int
    task: str
    code: str

    @computed_field
    @property
    def message(self) -> str:
        if self.success:
            return f'Agent successfully generated code for task {self.task} in {self.num_tries} tries!'
        else:
            return f'Agent unsuccessfully generated code for task {self.task} in {self.num_tries} :('


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
                 prompt_template_cls: BasePromptTemplate,
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
    def prompt_template(self, new_prompt_template: BasePromptTemplate) -> None:
        self._prompt_template = new_prompt_template

    def iteratively_generate(self, payload: StrategyInputPayload) -> Tuple[bool, str, StrategyOutputPayload, str]:
        '''Converts agent payload into validly formatted outputs iteratively'''

        success, tries = False, 0
        proxy_payload = deepcopy(payload)
        task = proxy_payload.task

        for tries in range(self.num_tries):

            success, llm_output, out_payload = self.generate(proxy_payload)
            message = self.message_cls(
                validation_success, tries, proxy_payload.task, out_payload.code)

            if success:
                if self.solver:
                    # NOTE: you do not have a solver, so this will not trigger
                    validation_success, validation_output = self.solver(
                        out_payload)
                    message = self.message_cls(
                        validation_success, tries, proxy_payload.task, out_payload.code)
                    if validation_success:
                        break
                    else:
                        proxy_payload.update_code_error(
                            validation_output.output, validation_output.error)
                        print(
                            f'Action agent output failed validation, re-prompting...') if self.verbose else None

            else:
                proxy_payload.update_code_error(llm_output, out_payload.error)
                print(
                    f'Action agent output in wrong format, re-prompting...') if self.verbose else None

        return success, llm_output, out_payload, message
