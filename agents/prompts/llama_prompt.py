"""
Class for constructing in Llama prompts for LLama-3 Vllm Chat Models 
"""
from __future__ import annotations
import json
import re
import os
import random
from typing import TypedDict, TypeVar, Optional, Any, Callable, Literal, Tuple
from textwrap import dedent
import itertools
import pprint as pp
from enum import Enum

from agents.utils import BaseConfig
from agents.prompts.base_prompt_template import BasePromptTemplate, BaseInputPayload, BaseOutputPayload
from agents import prompt_registry
from agents.prompts.gsm_llama_prompts import BASE

PROMPT_NAME = 'vllm'
Example = TypeVar('Example')
PARSE_TYPES = {
    'Q': re.compile(r"Question (\-?[0-9\.\,]+)"), 
    'A': re.compile(r"Answer (\-?[0-9\.\,]+)"), 
    'FA': re.compile(r"#### (\-?[0-9\.\,]+)")
}

class GSM8kPromptDict(TypedDict):
    """ Stores the Components for the Prompt """
    instruction: str
    interactive_examples: list[str]
    useful_examples: list[str]
    question_prefix: str
    subquestion_prefix: str
    overall_question_prefix: str
    answer_prefix: str
    
def read_jsonl(path: str) -> list[dict[str, str]]:
    """
    Reads jsonl and returns it 
    """
    with open(path) as fh:
        return [json.loads(line) for line in fh.readlines() if line]

def make_fsl_llama(prompt: GSM8kPromptDict, num_fsl_examples: int) -> list[dict[str, str]]:
    """ 
    Takes in a GSM8kPromptDict with a loaded instruction and fsl examples 
    and returns a list of {'user': ..., 'content': ...} messages for llama 
    """
    system_instruction = prompt['instruction']
    formatted_examples = []
    
    for idx, example_text in enumerate(prompt['interactive_examples']):
        formatted_example = [{"role": "system", "content": system_instruction}]
        # Extract the user question and answer pairs
        lines = example_text.strip().splitlines()
        # Add the main question to the formatted example
        main_question = next(line for line in lines if line.startswith("Question"))
        formatted_example.append({"role": "user", "content": main_question.strip().format(idx=idx + 1)})
        # Process sub-questions and answers
        for line in lines[1:]:
            if line.startswith("Question"):
                formatted_example.append({"role": "user", "content": line.strip().format(idx=idx + 1)})
            elif line.startswith("Answer"):
                formatted_example.append({"role": "assistant", "content": line.strip().format(idx=idx + 1)})
        formatted_examples.append(formatted_example)

    # Sample the specified number of formatted examples
    indices = random.sample(range(len(formatted_examples)), num_fsl_examples)
    selected_examples: list[list[dict]] = [formatted_examples[i] for i in indices]

    return list(itertools.chain(*selected_examples))

def postprocess_qa(llm_output: str) -> Tuple[bool, str | None, str | None]: 
    """ Post-processes llm_output into Question, Answer, or Final Answer"""
    success, match_type, parsed = False, None, None
    
    for name, re_obj in PARSE_TYPES.items(): 
        match = re_obj.search(llm_output)
        if match: 
            match_str = match.group(1).strip()
            match_str = match_str.replace(",", "")
            success, match_type, parsed = True, name, match_str
            
    return success, match_type, parsed

    
@prompt_registry.register(name=PROMPT_NAME)
class GSMLlamaPromptTemplate:
    """ Question answer prompt template."""

    def __init__(self, fsl_prompt_type: str = 'base', num_fsl_examples: int = 3) -> None:
        """Initialize the QuestionAnswerPromptTemplate."""
        
        FSL_PROMPTS = {
            'base': BASE
        }
        
        fsl_prompt_base: list[dict] = FSL_PROMPTS.get(fsl_prompt_type)
        if fsl_prompt_base is None: 
            raise ValueError(f'You must choose supported prompts: {FSL_PROMPTS.values()}')
        
        fsl_prompt_base: list[dict[str, str]] = make_fsl_llama(fsl_prompt_base, num_fsl_examples)
        
        self._fsl_prompt_base = fsl_prompt_base
        self.fsl_prompt = fsl_prompt_base
        
    def add(self, role: Literal['user', 'assistant', 'system'], content: str) -> None: 
        """ Add new content to the prompt """
        self.fsl_prompt.append({'role': role, 'content': content})
        
    def pop(self) -> None: 
        """ Pops the most recent interaction from prompt """
        self.fsl_prompt.pop(-1)
        
    def reset(self) -> None: 
        """ Resets the whole prompt to the base fsl prompt """
        self.fsl_prompt = self._fsl_prompt_base

    def preprocess(self, **kwargs) -> list[dict[str, str]]:
        """
        Preprocess the sequence of chats [{'u'}] if needed 
        and return as a list of dicts [{'user': ..., 'content': ...}, ...]

        Returns
        -------
        list[dict[str, str]]
            The formatted prompt. 
        """
        # Build the prompts using the template
        return self.fsl_prompt

    @staticmethod
    def postprocess(response: str, post_process_fn: Callable = postprocess_qa) -> Tuple[bool, str, str]:
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
        return post_process_fn(response)
    
    def __repr__(self) -> str:
        return pp.pformat(self.fsl_prompt)
    
    def __str__(self) -> str:
        return pp.pformat(self.fsl_prompt)
    
    
if __name__ == "__main__": 
    from agents.generators.vllm_generator import VLLMGenerator, VLLMGeneratorConfig
    from agents.prompt_breeder.gsm import batch_sample_qa_pairs
    
    prompt_template = GSMLlamaPromptTemplate()
    generator_cfg = VLLMGeneratorConfig()
    generator = VLLMGenerator(generator_cfg)
    
    dataset = read_jsonl('/lus/eagle/projects/FoundEpidem/bhsu/2024_research/agents/agents/data/gsm.jsonl')
    samples: list[dict] = batch_sample_qa_pairs(dataset, batch_size = 4)
    prompt_template.add('user', samples[0]['question'])
    
    for _ in range(10): 
        llm_output = generator.generate(prompt_template.preprocess())
        success, match_type, parsed  = prompt_template.postprocess(llm_output)
        if match_type == 'Q': 
            message = {'role': ''}
        elif match_type == 'A': 
            message = {''}
    
    breakpoint()