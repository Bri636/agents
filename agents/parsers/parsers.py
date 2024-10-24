from __future__ import annotations

''' Parsing Functions for Agents '''

from typing import Tuple, Literal, Any, Callable
import json

from enum import Enum

def strip_json(string: str) -> str:
    return string.strip('```json').strip('```').strip()

def load_json(string: str) -> str:
    return json.loads(string)

class LLMOutputParser:
    ''' Class that parses json outputs, then filters them through a custom output pydantic class 
    to return only the fields specified in the pydantic class. 
    '''

    def __init__(self, parser: Callable) -> None:
        self.parser = parser

    def __call__(self, llm_output: str) -> Tuple[bool, dict[str, str]]:
        '''Uses llm_output_parser to parse raw string output and then organize it into a ParsedOutput'''

        try:
            parsed_output: dict[str, Any] = self.parser(llm_output)
            parsed_output.update({'error': None})

            return True, parsed_output

        except Exception as e:
            parsed_output = {'error': e}
            # filled_payload: dict[str, None] = {
            #     k: None for k, _ in
            #     list(self.output_cls.model_fields.keys())}
            # parsed_output.update(filled_payload)

            return False, parsed_output
        
