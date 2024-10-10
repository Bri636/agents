"""Interface for all agents to follow."""

from __future__ import annotations

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

AGENT_NAME = 'Python'


class PythonAgentMessage(BaseModel):
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

@dataclass # I use dataclass since pydantic seems to cause error for non-basemodel classes
class PythonAgentContainer:
    ''' Config for running the python agent '''
    _name: str = AGENT_NAME
    generation_config: GenerationConfig = field(
        default=None,
        metadata='Config for how agent should generate code'
    )
    action_agent_cls: PythonCodeAgent = field(
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
    message_cls: PythonAgentMessage = field(
        default=None,
        metadata='Class for formatting agent messages'
    )
    
    def __post_init__(self) -> None: 
        
        for field_name, field_def in self.__dataclass_fields__.items(): 
            actual_field_value = getattr(self, field_name)
            expected_field_type = field_def.type
            # if type(actual_field_value) != expected_field_type: 
            #     return TypeError(f'''You provided wrong class {actual_field_value} for {field_name}...
            #                      the class type is supposed to be {expected_field_type}''')
   
            # if field_name=='prompt_template_cls':
            #     print('Test')
            #     object.__setattr__(self, field_name, self.prompt_template_cls())
                
@action_agent_registry.register(AGENT_NAME,
                                cls_container=PythonAgentContainer,
                                cls_payload={
                                    'generation_config': GenerationConfig,
                                    'input_payload_cls': PythonInputPayload,
                                    'prompt_template_cls': PythonPrompt,
                                    'output_payload_cls': PythonOutputPayload,
                                    'message_cls': PythonAgentMessage
                                })
class PythonCodeAgent(BaseActionAgent):
    ''' Very basic python coding agent '''

    def __init__(self,
                 generation_config: GenerationConfig,
                 generator: BaseLLMGenerator,
                 prompt_template_cls: BasePromptTemplate,
                 llm_output_parser: Optional[Union[LLMOutputParser, Callable]],
                 solver: Optional[Union[Callable]],
                 output_payload_cls: PythonOutputPayload,
                 message_cls: PythonAgentMessage,
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
