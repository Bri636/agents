from __future__ import annotations

''' Parsing Functions for Agents '''

from typing import Tuple
import json

from enum import Enum

def strip_json(string: str) -> str: 
    return string.strip('```json').strip('```').strip()

def load_json(string: str) -> str: 
    return json.loads(string)

OUTPUT_FORMATS = {
    'json_dict': load_json, 
    'json_str': strip_json
}

class JsonParser: 
    
    def __init__(self, output_format_strategy: str) -> None:
        
        output_func = OUTPUT_FORMATS.get(output_format_strategy)
        self.output_func = output_func
        
    



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