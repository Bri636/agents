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
    
    def generate_answer(self, idx: int, sample: GSM8KProblem, num_tries: int) -> Tuple[bool, str | None]:
        """
        Performs eval on one question from GSM8K and returns if successfully parsed, 
        and if llm was correct or not 
        """
        
        question: str = sample['question']
        self.prompt.add(**{'role': 'user', 'content': question})
        
        try: 
            for _ in range(num_tries): 
                
                llm_output = self.generator.generate(self.prompt.preprocess())[0]
                
                filtered_output = self.llm_output_filter(llm_output)
                if filtered_output == 'final_answer': 
                    self.prompt.reset()
                    out = gsm_is_correct(idx, llm_output, sample)
                    return True, out
                elif filtered_output == '[invalid]': 
                    pass 
                
            return True, False # else, return best guess
                
        except Exception as e: 
            return False, None
        
    def reset_pass(self) -> None: 
        self.prompt.reset()
        
        
if __name__=="__main__": 
    import random
    from agents.generators.argo_chat_generator import LangChainFSLGenerator, ArgoGeneratorConfig
    from agents.generators.vllm_generator import VLLMGenerator, VLLMGeneratorConfig
    from agents.prompts.standard_prompt import StandardGSMPromptTemplate
    from agents.gsm8k.utils import read_jsonl
    from tqdm import tqdm
    
    dataset = read_jsonl('/Users/BrianHsu/Desktop/GitHub/agents/agents/data/gsm.jsonl')
    
    # cfg = ArgoGeneratorConfig()
    cfg = VLLMGeneratorConfig()
    # generator = LangChainFSLGenerator(cfg)
    generator = VLLMGenerator(cfg)
    prompt = StandardGSMPromptTemplate()
    
    reasoner = LLMReasoner(generator, prompt, filter_output_type)
    num_samples = 30
    num_tries = 10
    
    sample_idx = random.sample(range(len(dataset)), num_samples)
    samples = [dataset[i] for i in sample_idx]
    
    num_correct = 0
    num_completed = 0
    for idx, sample in tqdm(enumerate(samples)): 
        finished, correct = reasoner.generate_answer(idx, sample, num_tries)
        num_correct += finished and correct
        num_completed += finished
        reasoner.reset_pass()

        print(f'\nCorrect: {num_correct}')
    percent_completed = (num_completed / num_samples) * 100
    percent_correct = (num_correct / num_completed) * 100
        
    print(f'Percent Completed: {percent_completed}\nPercent Correct: {percent_correct}')
    # final_answer = reasoner.generate_answer(num_tries=10, sample=samples[0])

    breakpoint()
    