""" Container class that uses two question-answer agents """

from __future__ import annotations
from typing import Callable, Tuple

from agents.reasoners.base_reasoner import BaseReasoner
from agents.utils import BaseConfig
from agents.gsm8k.utils import filter_output_type, gsm_is_correct
from agents.generators import BaseLLMGenerator
from agents.prompts import BasePromptTemplate

from agents.gsm8k import GSM8KProblem
    
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
    
    def generate_answer(self, num_tries: int, sample: GSM8KProblem) -> Tuple[bool, str | None]:
        """ Within number of tries, generate answer in proper format """
        
        question: str = sample['question']
        self.prompt.add(question)
        
        try: 
            for _ in num_tries: 
                
                llm_output = self.generator.generate(self.prompt.preprocess())
                output_type = self.llm_output_filter(llm_output)
                
                if output_type == 'final_answer': 
                    self.prompt.reset()
                    return True, llm_output
                elif output_type == '[invalid]': 
                    pass 
                
        except Exception as e: 
            return False, None
    