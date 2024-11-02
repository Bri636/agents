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
from agents.prompts.gsm_llama_prompts import BASE, QUESTION, ANSWER

PROMPT_NAME = 'vllm'
Example = TypeVar('Example')

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

def make_fsl_llama(prompt: GSM8kPromptDict, num_fsl_examples: int, agent_type: Literal['question', 'answer']) -> list[dict[str, str]]:
    """ 
    Takes in a GSM8kPromptDict with a loaded instruction and fsl examples 
    and returns a list of {'user': ..., 'content': ...} messages for llama 
    """
    system_instruction = prompt['instruction']
    formatted_examples = []
    
    # if question agent, ai answers are the questions
    if agent_type == 'question': 
        q_role = 'assistant'
        a_role = 'user'
    else: 
        q_role = 'user'
        a_role = 'assistant'
    
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
                formatted_example.append({"role": q_role, "content": line.strip().format(idx=idx + 1)})
            elif line.startswith("Answer"):
                formatted_example.append({"role": a_role, "content": line.strip().format(idx=idx + 1)})
        formatted_examples.append(formatted_example)

    # Sample the specified number of formatted examples
    indices = random.sample(range(len(formatted_examples)), num_fsl_examples)
    selected_examples: list[list[dict]] = [formatted_examples[i] for i in indices]

    return list(itertools.chain(*selected_examples))
    
@prompt_registry.register(name=PROMPT_NAME)
class GSMLlamaPromptTemplate:
    """ Question answer prompt template."""

    def __init__(self, fsl_prompt_type: str = 'question', num_fsl_examples: int = 1, agent_type: str = None) -> None:
        """Initialize the QuestionAnswerPromptTemplate."""
        
        FSL_PROMPTS = {
            'base': BASE,
            'question': QUESTION, 
            'answer': ANSWER
        }
        
        fsl_prompt_base: list[dict] = FSL_PROMPTS.get(fsl_prompt_type)
        if fsl_prompt_base is None: 
            raise ValueError(f'You must choose supported prompts: {FSL_PROMPTS.values()}')
        
        fsl_prompt_base: list[dict[str, str]] = make_fsl_llama(fsl_prompt_base, num_fsl_examples, agent_type)
        
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
    
    def __repr__(self) -> str:
        return pp.pformat(self.fsl_prompt)
    
    def __str__(self) -> str:
        return pp.pformat(self.fsl_prompt)
    

def filter_output_type(llm_output: str) -> Literal['question', 'answer', 'final_answer', '[invalid]']: 
    """ Filter an llm output and returns what kind of response it is """
    Q = re.compile(r"Question (\-?[0-9\.\,]+)")
    A = re.compile(r"Answer (\-?[0-9\.\,]+)")
    FA = re.compile(r"####")
    
    if Q.search(llm_output): 
        return 'question'
    elif A.search(llm_output): 
        return 'answer'
    elif FA.search(llm_output): 
        return 'final_answer'
    else: 
        return '[invalid]'
    
def gsm_extract_answer(completion: str) -> str:
    """ 
    Parses through a string and returns the answer as a str
    
    Expects the answer in this format: 
    Answer is #### -567.89 ===> -567.89
    """
    ANS_RE = re.compile(r"#### (\-?[0-9\.\,]+)")
    INVALID_ANS = "[invalid]"
    
    match = ANS_RE.search(completion)
    if match:
        match_str = match.group(1).strip()
        match_str = match_str.replace(",", "")
        return match_str
    else:
        return INVALID_ANS
    
def gsm_is_correct(answer: str, gold_answer: dict[str, str]) -> bool: 
    return bool(
        float(gsm_extract_answer(answer)) == float(gsm_extract_answer(gold_answer["answer"]))
        )
    
if __name__ == "__main__": 
    
    from agents.generators.vllm_generator import VLLMGenerator, VLLMGeneratorConfig
    from agents.prompt_breeder.gsm import batch_sample_qa_pairs
    
    q_prompt: GSMLlamaPromptTemplate = GSMLlamaPromptTemplate('question', 1, 'question')
    a_prompt: GSMLlamaPromptTemplate = GSMLlamaPromptTemplate('answer', 1, 'answer')
    generator_cfg = VLLMGeneratorConfig()
    generator = VLLMGenerator(generator_cfg)
    
    dataset = read_jsonl('/lus/eagle/projects/FoundEpidem/bhsu/2024_research/agents/agents/data/gsm.jsonl')
    samples: list[dict] = batch_sample_qa_pairs(dataset, batch_size = 1)
    problem = samples[0]['question']
    answer = samples[0]['answer']
    
    q_prompt.add('user', content=problem)
    a_prompt.add('user', content=problem)
    
    breakpoint()
    
    for _ in range(10): 
        sub_q = generator.generate(q_prompt.preprocess())[0]
        q_prompt.add(**{'role': 'assistant', 'content': sub_q})
        a_prompt.add(**{'role': 'user', 'content': sub_q})
        sub_a = generator.generate(a_prompt.preprocess())[0]
        q_prompt.add('user', sub_a)
        a_prompt.add('assistant', sub_a)
        
        if filter_output_type(sub_a) == 'final_answer': 
            break
        
    out = gsm_is_correct(sub_a, samples[0])
    breakpoint()

    
    