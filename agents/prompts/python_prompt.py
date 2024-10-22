from __future__ import annotations

from typing import Any, Union, Optional
from textwrap import dedent
from pydantic import Field
import json

""" Basic Python Prompt Implementation """

from agents.configs.configs import BaseConfig
from agents.prompts.base_prompt import BasePromptTemplate, BaseOutputPayload, BaseInputPayload

from agents import prompt_registry
from agents import input_payload_registry
from agents import output_payload_registry

AGENT_NAME = 'Python' # name of the agent 

class PythonOutputPayload(BaseOutputPayload): 
    
    reasoning: str = Field(
        description='The reasoning for the task that is given to you. ',
        default='None'
    )
    
    code: str = Field(
        description='Your python function as a string.',
        default='None'
    ) 

class PythonInputPayload(BaseInputPayload): 
    
    strategy: str = Field(
        description='string strategy to use', 
        default='Think Step by Step'
    )
    
    output_format: str = Field(
        description='Output format json from OutputPayload', 
        default=PythonOutputPayload.get_output_format()
    )
    
    task: str = Field(
        description='The task to complete', 
        default='None'
    )
    
    code_functions: str = Field(
        description='Text of code functions', 
        default='None'
    )
    
    previous_attempt: str = Field(
        description='Previous attempt at code', 
        default='None'
    )
    
    error: str = Field(
        description='error message from code checker', 
        default='None'
    )
    
    def update_code_error(self, llm_output: str, error: str) -> None: 
        ''' Adds an error feedback to the input payload for re-prompting the model '''
        
        self.previous_attempt = llm_output
        self.error = error
    
class PythonPrompt(BasePromptTemplate): 
    
    template: str = dedent('''
System: 
You are an intelligent coding agent that is an expert in running python workflows. 
Your goal is to help me execute python code that I will give you based on a task I specify. 

At each round of conversation, I will provide you with the following: 

Task: 
// This is a specific instruction that I will give you that you will have to turn into code in json format. //

Code Functions: 
// Python functions that I currently have, along with their descriptions // 

Previous Attempt: 
// Your previous attempt at calling code for the task, where you failed in some way. //

Error: 
// The error corresponding to your previous attempt. //

Here are some strategies for how you should think about coding: 
{strategy}

You will respond to me in the following format: 
Output Format: 
{output_format}

Human: 

Task: 
{task}

Code Functions: 
{code_functions}

Previous Attempt: 
{previous_attempt}

Error: 
{error}
    ''')
    

