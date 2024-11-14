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
import copy

from agents.utils import BaseConfig
from agents.prompts.base_prompt_template import BasePromptTemplate
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
    

class GSMLlamaPromptTemplate(BasePromptTemplate):
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
        
        self._fsl_prompt_base = copy.deepcopy(fsl_prompt_base)
        self.fsl_prompt = copy.deepcopy(fsl_prompt_base)
        self._history = []  # Initialize the change history
        self._fsl_prompt_type = fsl_prompt_type
        
    @property
    def prompt_type(self) -> str: 
        return self._fsl_prompt_type
    
    @property
    def history(self) -> list[dict[str, str]]:
        """Returns a copy of the change history to prevent in-place modification."""
        return copy.deepcopy(self._history)
        
    def add(self, role: Literal['user', 'assistant', 'system'], content: str) -> None: 
        """ Add new content to the prompt """
        self.fsl_prompt.append({'role': role, 'content': content})
        self._history.append({'role': role, 'content': content})  # Track the addition
        
    def pop(self, indices: list[int] = [-1]) -> None:
        """Removes specified indices from the prompt and also adjusts change history."""
        indices_set = {i if i >= 0 else len(self.fsl_prompt) + i for i in indices}
        indices_set = {i for i in indices_set if 0 <= i < len(self.fsl_prompt)}
        # Track the items to be removed
        removed_items = [item for idx, item in enumerate(self.fsl_prompt) if idx in indices_set]
        # Remove these items from both fsl_prompt and _change_history if they are in _change_history
        self.fsl_prompt = [item for idx, item in enumerate(self.fsl_prompt) if idx not in indices_set]
        self._history = [item for item in self._history if item not in removed_items]
        
    def reset(self) -> None: 
        """ Resets the whole prompt to the base fsl prompt """
        self.fsl_prompt = copy.deepcopy(self._fsl_prompt_base)
        self._history = []  # Clear the change history

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
    
    def copy_history(self, prompt: GSMLlamaPromptTemplate) -> None:
        """
        Sets the current prompt and history based on another GSMLlamaPromptTemplate instance.
        If the current prompt type is the opposite of the input prompt type, swaps 'user' and 'assistant' roles
        in the copied history and appends the history to the current fsl_prompt.
        
        Parameters
        ----------
        prompt : GSMLlamaPromptTemplate
            The instance from which to copy the prompt and history.
        """
        if not isinstance(prompt, GSMLlamaPromptTemplate):
            raise TypeError("Prompt must be an instance of GSMLlamaPromptTemplate.")
        breakpoint()
        # Determine if we need to swap roles by checking if the prompt types are different
        if self._fsl_prompt_type != prompt._fsl_prompt_type:
            # Swap roles in the copied history if prompt types are opposite
            modified_history = [
                {'role': 'assistant' if entry['role'] == 'user' else 'user' if entry['role'] == 'assistant' else entry['role'],
                'content': entry['content']}
                for entry in prompt._history
            ]
        else:
            # Directly copy the history if no swapping is needed
            modified_history = copy.deepcopy(prompt._history)
        
        # Update self._history and append to self.fsl_prompt
        self._history = modified_history
        self.fsl_prompt.extend(modified_history)

if __name__ == "__main__": 
    
    from agents.generators.vllm_generator import VLLMGenerator, VLLMGeneratorConfig
    from agents.gsm8k.utils import batch_sample_gsm, filter_output_type, gsm_is_correct
    
    def log_prob_reward(log_probs_seq: list[float]) -> float: 
        """ Returns the average log probability"""
        return float(sum(log_probs_seq) / len(log_probs_seq))
    
    q_prompt: GSMLlamaPromptTemplate = GSMLlamaPromptTemplate('question', 1, 'question')
    a_prompt: GSMLlamaPromptTemplate = GSMLlamaPromptTemplate('answer', 1, 'answer')
    
    breakpoint()
    q_prompt.add(**{'role': 'user', 'content': "TESTING"})
    q_prompt.add(**{'role': 'assistant', 'content': "OTHER_TESTING"})
    breakpoint()
    
    test_prompt = GSMLlamaPromptTemplate('answer', 1, 'answer')
    test_prompt.copy_history(q_prompt)
    
    q_prompt.pop([-1, -2])
    breakpoint()
    
    generator_cfg = VLLMGeneratorConfig()
    generator = VLLMGenerator(generator_cfg)
    
    dataset = read_jsonl('/lus/eagle/projects/FoundEpidem/bhsu/2024_research/agents/agents/data/gsm.jsonl')
    samples: list[dict] = batch_sample_gsm(dataset, batch_size = 1)
    problem = samples[0]['question']
    answer = samples[0]['answer']
    
    q_prompt.add('user', content=problem)
    a_prompt.add('user', content=problem)
    
    breakpoint()
    
    for _ in range(10): 
        sub_q_dict = generator.generate(q_prompt.preprocess())
        q_prompt.add(**{'role': 'assistant', 'content': sub_q_dict['text'][0]})
        a_prompt.add(**{'role': 'user', 'content': sub_q_dict['text'][0]})
        
        sub_a_dict = generator.generate(a_prompt.preprocess())
        q_prompt.add('user', sub_a_dict['text'][0])
        a_prompt.add('assistant', sub_a_dict['text'][0])
        
        if filter_output_type(sub_a_dict['text'][0]) == 'final_answer': 
            break
        
    out = gsm_is_correct(sub_a_dict['text'][0], samples[0])
    breakpoint()

    
    