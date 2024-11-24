""" Container class that uses two question-answer agents """

from __future__ import annotations
from typing import Callable, Tuple
from rich.panel import Panel

from agents.reasoners.base_reasoner import BaseReasoner
from agents.utils import BaseConfig
from agents.gsm8k.utils import filter_output_type, gsm_is_correct
from agents.generators import BaseLLMGenerator
from agents.prompts import BasePromptTemplate
from agents.gsm8k import GSM8KProblem
from agents.prompts.standard_prompt import StandardGSMPromptTemplate

from agents.reasoners import T

@BaseReasoner.register(name='base')
class LLMReasoner(BaseReasoner): 
    """ Base LLM Class for one-shot reasoning """
    def __init__(self, 
                 generator: BaseLLMGenerator, 
                 prompt: BasePromptTemplate, 
                 llm_output_filter: Callable = filter_output_type
                 ) -> None:
        """ 
        Naive llm used for generating answers for GSM8K questions
        
        :param generator: BaseLLMGenerator - LLM Generator 
        :param prompt: BasePromptTemplate - Prompt Template
        :param llm_output_filter: Callable - Callable used to filter for what type of output an llm is giving: question, answer, final_answer, or invalid output
        """
        
        self.generator = generator
        self.prompt = prompt
        self.llm_output_filter = llm_output_filter
    
    def generate_answer(self, 
                        idx: int, 
                        sample: GSM8KProblem, 
                        num_tries: int
                        ) -> Tuple[bool, bool, str, Panel | None]:
        """
        Performs eval on one question from GSM8K and returns if successfully parsed, 
        and if llm was correct or not 
        """
        
        question: str = sample['question']
        self.prompt.add(**{'role': 'user', 'content': question})
        
        generated, correct = False, False
        message, panel = f'Answer Incorrect or failed to Generate for Question :(', None
        try: 
            for _ in range(num_tries): 
                
                llm_output = self.generator.generate(self.prompt.preprocess())[0]
                
                filtered_output = self.llm_output_filter(llm_output)
                if filtered_output == 'final_answer': 
                    self.prompt.reset()
                    correct, message = gsm_is_correct(idx, llm_output, sample)
                    return True, correct, message, panel
                
                elif filtered_output == '[invalid]': 
                    pass 
                
            return True, False, message, panel # else, return best guess
                
        except Exception as e: 
            return False, False, message, panel
        
    def reset_pass(self) -> None: 
        self.prompt.reset()
        
    @classmethod
    def initialize(cls: T, 
                   generator: BaseLLMGenerator, 
                   filter_output_func: Callable = filter_output_type
                   ) -> T:
        prompt = StandardGSMPromptTemplate()
        return cls(generator, prompt, filter_output_func)
    
    