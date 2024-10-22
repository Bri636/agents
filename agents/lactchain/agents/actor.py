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

from agents.configs import BaseConfig
from agents.generators.base_generator import BaseLLMGenerator
from agents.prompts.base_prompt import BasePromptTemplate
from agents.llm_agents.base_action_agent import BaseActionAgent

from agents.prompts.python_prompt import PythonInputPayload, PythonPrompt, PythonOutputPayload
from agents.parsers import LLMOutputParser

from agents import action_agent_registry

T = TypeVar('T')

AGENT_NAME = 'LactChainActor'

_T = TypeVar('_T')


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
class LactChainActorAgentContainer:
    ''' Config for running the python agent '''
    _name: str = AGENT_NAME
    generation_config: GenerationConfig = field(
        default=None,
        metadata='Config for how agent should generate code'
    )
    action_agent_cls: LactChainActorChain = field(
        default=None,
        metadata=' Action Agent cls '
    )
    generator: BaseLLMGenerator = field(
        default=None,
        metadata='llm generator for agent'
    )
    prompt_template_cls: PythonPrompt = field(
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
    input_payload_cls: PythonInputPayload = field(
        default=None,
        metadata='Payload container that forms inputs for the agent'
    )
    output_payload_cls: PythonOutputPayload = field(
        default=None,
        metadata='Payload container that forms outputs from the agent'
    )
    message_cls: LactChainAgentMessage = field(
        default=None,
        metadata='Class for formatting agent messages'
    )

    def __post_init__(self) -> None:
        pass

class LactChainActorChain(BaseActionAgent):
    '''Single Actor-Based Chain for On-Policy Sampling
    For Now: 
    ========

    Initialize: Strategy + Parser
    Input: observation + info
    Output: sequence of actions
    '''

    def __init__(self,
                 generation_config: GenerationConfig,
                 generator: BaseLLMGenerator,
                 prompt_template_cls: BasePromptTemplate,
                 llm_output_parser: Optional[Union[LLMOutputParser, Callable]],
                 solver: Optional[Union[Callable]],
                 output_payload_cls: PythonOutputPayload,
                 message_cls: Optional[LactChainAgentMessage],
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
        
    def generate(self, payload: PythonInputPayload) -> Tuple[bool, str, PythonOutputPayload]:
        '''Function that runs llm with payload and promtpt, then parses it and returns parsing result
        with success flag as bool...For now, just grab first element from list

        Returns: 
        =======
        success: bool 
            Measures whether llm successfully parsed or not 
        output: str
            Raw output of llm 
        parsed: dict[str, Any]
            Parsed code or {'error': Exception message}
        '''

        prompt = self.prompt_template.preprocess(**payload.model_dump())
        llm_output = self.generator.generate(prompt)[0]
        success, parsed = self.llm_output_parser(llm_output)
        out_payload = self.output_payload_cls(**parsed)

        return success, llm_output, out_payload
    
    def iteratively_generate(self, payload: PythonInputPayload) -> Tuple[bool, str, PythonOutputPayload, str]:
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

    def map_actions(self, **kwargs) -> None:
        """ Optional that maps parsed outputs to executable actions that can be executed in the environment """
        pass

    def execute(self, out_payload: PythonOutputPayload) -> list[Any]:
        """ Optional that executes mapped actions """
        pass


# class ActorChain(ActorChain):
#     '''Single Actor-Based Chain for On-Policy Sampling
#     For Now: 
#     ========

#     Initialize: Strategy + Parser
#     Input: observation + info
#     Output: sequence of actions
#     '''

#     def __init__(self,
#                  generator: LLMGenerator,
#                  prompt_template: ActionPromptTemplate,
#                  solver: Optional[ActionSolver] = None,
#                  environment_processor: Optional[Callable] = None
#                  ) -> None:
#         '''Container actor chain class'''

#         self.generator = generator
#         self.prompt_template = prompt_template
#         self.solver = solver
#         self.environment_processor = environment_processor

#     def _preprocess(self,
#                     strategies: str | list[str],
#                     states: str | list[str],
#                     infos: Optional[str | list[str]] = None
#                     ) -> list[str]:
#         '''Preprocesses strings into prompt templates'''

#         if self.environment_processor:
#             states, infos = self.environment_processor(states, infos)

#         prompts = self.prompt_template.preprocess(strategy=strategies,
#                                                   state=states,
#                                                   info=infos)

#         return prompts

#     def parse_outputs(self, outputs: list[str]) -> Tuple[list[str], list[str]]:
#         '''Loops through the list of outputs and json parses them to return a list of 
#         processed strings
#         '''
#         parsed_explanations = []
#         parsed_moves = []
#         for _, output in enumerate(outputs):
#             parsed_explanations.append(json.loads(output)['explain'])
#             parsed_moves.append(json.loads(output)['moves'])

#         return parsed_explanations, parsed_moves

#     def map_actions(self, batch_moves: list[str]) -> list[np.ndarray]:
#         '''Helper function that maps list of processed outputs from llm into binary actions 
#         as a list of arrays
#         '''
#         batch_mapped_actions = self.solver.convert(batch_moves)

#         return batch_mapped_actions

#     def sample_actions(self,
#                        strategies: str | list[str],
#                        states: str | list[str],
#                        infos: str | list[str]
#                        ) -> list[np.ndarray]:
#         '''Samples batches of actions as List[array(int={0, 1})] where list is the batch, 
#         array is the compound actions'''

#         prompts = self._preprocess(strategies=strategies,
#                                    states=states,
#                                    infos=infos)

#         outputs = self.generator.generate(prompts)
#         explanations, moves = self.parse_outputs(outputs)
#         actions = self.map_actions(moves)

#         return actions
