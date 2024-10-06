from __future__ import annotations

''' Parsing Functions for Agents '''

from typing import Tuple, Literal, Any, Callable
import json

from enum import Enum

from agents.prompts.base_prompt import BaseOutputPayload

def strip_json(string: str) -> str:
    return string.strip('```json').strip('```').strip()

def load_json(string: str) -> str:
    return json.loads(string)

class LLMOutputParser:
    ''' Class that parses json outputs, then filters them through a custom output pydantic class 
    to return only the fields specified in the pydantic class. 
    '''

    def __init__(self, output_cls: BaseOutputPayload, parser: Callable) -> None:

        self.output_cls = output_cls
        self.parser = parser

    def parse_from_output(self, llm_output: str) -> BaseOutputPayload:
        '''Uses llm_output_parser to parse raw string output and then organize it into a ParsedOutput'''

        try:
            parsed_output: dict[str, Any] = self.parser(llm_output)
            parsed_output.update({'error': None})

            return self.output_cls(**parsed_output)

        except Exception as e:
            parsed_output = {'error': e}
            filled_payload: dict[str, None] = {
                k: None for k, _ in
                list(self.output_cls.model_fields.keys())}
            parsed_output.update(filled_payload)

            return self.output_cls(**parsed_output)


def parse_output(output: str) -> Tuple[bool, dict[str, str]]: 
    '''General function that parses output of llm via different json methods
    Input: 
    =====
    output: str
        json output of llm 
    
    Output: 
    ======
    success: bool 
        if successfully parsed
    parsed_output: dict
        dict that holds llm output, else a dict that contains error
    '''
    
    try:
        parsed_output = json.loads(output)
        return (True, parsed_output)  
    
    except Exception as e: 
        try: 
            parsed_output = json.loads(output[1:-1])
            return (True, parsed_output)  
         
        except Exception as e: 
            try: 
                stripped_output = output.strip('```json').strip('```').strip()
                parsed_output = json.loads(stripped_output)
                return (True, parsed_output)
            
            except Exception as e:
                parsed_output = {'error': e}
                return (False, {'error': parsed_output})