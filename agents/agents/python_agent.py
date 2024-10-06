"""Interface for all agents to follow."""

from __future__ import annotations

from typing import Protocol, Callable, Optional, Any, TypeVar, Type, Union, runtime_checkable, Tuple
from abc import ABC, abstractmethod
from copy import deepcopy

from agents.configs import BaseConfig
from agents.base_classes import BaseLLMGenerator
from agents.base_classes import BasePromptTemplate
from agents.base_action_agent import BaseActionAgent

from agents.prompts.python_prompt import PythonInputPayload
from agents.parsers import LLMOutputParser
from agents import action_agent_registry

T = TypeVar('T')

AGENT_NAME = 'Python'

class PythonAgentConfig(BaseConfig): 
    ''' Config for running the python agent '''
    
    _name: str = AGENT_NAME
    verbose: bool = False
    num_tries: int = 5
    

action_agent_registry.register(AGENT_NAME)
class PythonCodeAgent(BaseActionAgent): 
    
    def __init__(self, 
                config: BaseConfig, 
                generator: BaseLLMGenerator,
                prompt_template: BasePromptTemplate,
                llm_output_parser: Optional[Union[Type, Callable]], 
                solver: Optional[Union[Type, Callable]], 
                **kwargs
                ) -> None:
        
        self._generator = generator
        self._prompt_template = prompt_template
        
        self.llm_output_parser = llm_output_parser
        self.solver = solver
        
        self.verbose = config.verbose
        self.num_tries = config.num_tries
        
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
        
    def generate(self, payload: PythonInputPayload) -> Tuple[bool, str, dict[str, Any]]:
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
        output = self.generator.generate(prompt)[0]
        success, parsed = self.llm_parser(output)

        return success, output, parsed
    
    
    def iteratively_generate(self, payload: PythonInputPayload) -> Tuple[bool, dict[str, Any], str]:
        '''Converts agent payload into validly formatted outputs iteratively'''

        success, tries = False, 0
        proxy_payload = deepcopy(payload)
        robot = proxy_payload.robot
        task = proxy_payload.task
        
        for tries in range(self.num_tries): 

            success, output, parsed = self._generate(proxy_payload)
            
            if success:
                validation_success, validation_output = self.validator.validate(parsed['code'], robot)
                
                if validation_success: 
                    message = self.format_success(self.agent_message, tries+1, parsed)
                    break
                else: 
                    message = self.format_fail(self.agent_message, tries, task)
                    proxy_payload.add_error(output, validation_output)
                    
                    print(f'Action agent output failed validation, re-prompting...') if self.verbose else None
            
            else: 
                message = self.format_fail(self.agent_message, tries+1, task)
                proxy_payload.add_error(output, parsed)
                
                print(f'Action agent output in wrong format, re-prompting...') if self.verbose else None
                
        return success, output, parsed, message
         
    def preprocess(self, **kwargs) -> str: 
        """Preprocesses raw input with prompt_template and returns a string"""
        pass
        
    def parse_outputs(self, **kwargs) -> list[Any]: 
        """ Loops through raw outputs and parses it with the parser """
        pass
    def map_actions(self, **kwargs) -> list[Any]: 
        """ Optional that maps parsed outputs to actions that can be executed in the environment """
        pass
    
    

    