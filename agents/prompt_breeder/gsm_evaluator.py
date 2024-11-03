""" Evaluator class for evaluating how well the llm agents perform """

from __future__ import annotations
from typing import Callable, Tuple, Any
import random

from agents.prompt_breeder.gsm import read_jsonl
from agents.generators.vllm_generator import VLLMGenerator, VLLMGeneratorConfig
from agents.prompts.llama_prompt import GSMLlamaPromptTemplate
from agents.prompt_breeder.gsm import filter_output_type, gsm_is_correct

class GSMEvaluator: 

    def __init__(self, 
                 dataset: list[dict[str, str]],
                 generator: VLLMGenerator, 
                 question_prompt: GSMLlamaPromptTemplate,
                 answer_prompt: GSMLlamaPromptTemplate, 
                 parser: Callable
                 ) -> None:
        
        self.dataset = dataset
        self.generator = generator
        self.question_prompt = question_prompt
        self.answer_prompt = answer_prompt
        self.parser = parser
    
    def single_pass(self, sample: dict[str, str], num_tries: int) -> Tuple[bool, bool | None]: 
        """ 
        Performs eval on one question from GSM8K and returns if successfully parsed, 
        and if llm was correct or not
        """
        question = sample['question']
        self.question_prompt.add('user', content=question)
        self.answer_prompt.add('user', content=question)
        
        for _ in range(num_tries): 
            sub_q = self.generator.generate(self.question_prompt.preprocess())[0]
            self.question_prompt.add(**{'role': 'assistant', 'content': sub_q})
            self.answer_prompt.add(**{'role': 'user', 'content': sub_q})
            
            sub_a = self.generator.generate(self.answer_prompt.preprocess())[0]
            self.question_prompt.add('user', sub_a)
            self.answer_prompt.add('assistant', sub_a)
        
            if filter_output_type(sub_a) == 'final_answer': 
                out = gsm_is_correct(sub_a, sample)
                return True, out
            
        return False, None
    
    def reset_pass(self) -> None: 
        """ Resets the prompts """
        self.question_prompt.reset()
        self.answer_prompt.reset()
    
    def evaluate(self, num_samples: int = 100, num_tries: int = 10) -> dict[str, Any]:
        """ Performs evaluation on N samples from GSM8K within M tries, and calculates the metrics for them """
        sample_idx = random.sample(range(len(self.dataset)), num_samples)
        samples = [self.dataset[i] for i in sample_idx]
        
        num_correct = 0
        num_completed = 0
        for _, sample in enumerate(samples): 
            
            finished, correct = self.single_pass(sample, num_tries)
            num_correct += finished and correct
            num_completed += finished
            self.reset_pass()
            
        percent_completed = (num_completed / num_samples) * 100
        percent_correct = (num_correct / num_completed) * 100
        
        return {'completed': percent_completed, 'correct': percent_correct}
            
            
        
if __name__ == "__main__": 
    
    from agents.generators.vllm_generator import VLLMGenerator, VLLMGeneratorConfig
    from agents.prompt_breeder.gsm import batch_sample_qa_pairs, filter_output_type, gsm_is_correct
    
    def log_prob_reward(log_probs_seq: list[float]) -> float: 
        """ Returns the average log probability"""
        return float(sum(log_probs_seq) / len(log_probs_seq))
    
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
    