""" Reasoner that uses Two agents: world model agent and action agent to reason through a problem """

from __future__ import annotations
from typing import Callable, Any, Tuple, Optional, Self
from rich.panel import Panel

from agents.prompts.llama_prompt import GSMLlamaPromptTemplate
from agents.generators import BaseLLMGenerator
from agents.generators.vllm_generator import VLLMGenerator
from agents.reasoners.base_reasoner import BaseReasoner
from agents.prompts import BasePromptTemplate
from agents.gsm8k.utils import filter_output_type, gsm_is_correct
from agents.gsm8k.types import GSM8KProblem

class WorldModel: 
    def __init__(self, generator: BaseLLMGenerator) -> None:
        
        self.generator = generator

    def step(self, answer_prompt: BasePromptTemplate) -> str: 
        """ Generates the next state given the current action or state """
        sub_answer = self.generator.generate(answer_prompt.preprocess())[0]
        
        return sub_answer
    
    def step_logprobs(self, answer_prompt: BasePromptTemplate) -> dict: 
        """ Returns the next sub_question to ask"""
        assert isinstance(self.generator, VLLMGenerator), f"""
        LogProbs only supported with VLLM for now...
        """
        sub_answer = self.generator.generate_with_logprobs(answer_prompt.preprocess())
        return {'text': sub_answer['text'][0],
                'log_probs': sub_answer['log_probs'],
                }
        
    def is_terminal(self): 
        ...
        
    def _generate_reward(self): 
        """ Generates the reward for an action """
        
    def _generate_next_state(self): 
        """ Generates the next state in the environment for an action """
        
    def reset(self): 
        """ Resets the environment to its original state"""
        ...
        
    def prompt_exceeds_limit(self, prompts: BasePromptTemplate): 
        return self.generator.prompt_exceeds_limit(prompts.preprocess())
    
class Actor: 
    def __init__(self, 
                 generator: BaseLLMGenerator, 
                 ) -> None:
        
        self.generator = generator
    
    def act(self, question_prompt: BasePromptTemplate) -> str: 
        """ Returns the next sub_question to ask"""
        sub_question = self.generator.generate(question_prompt.preprocess())[0]
        
        return sub_question
    
    def act_logprobs(self, question_prompt: BasePromptTemplate) -> dict: 
        """ Returns the next sub_question to ask"""
        assert isinstance(self.generator, VLLMGenerator), f"""
        LogProbs only supported with VLLM for now...
        """
        sub_question = self.generator.generate_with_logprobs(question_prompt.preprocess())
        return {'text': sub_question['text'][0],
                'token_seq': sub_question['token_seq'],
                'log_probs': sub_question['log_probs'],
                }
         
    def prompt_exceeds_limit(self, prompts: BasePromptTemplate): 
        return self.generator.prompt_exceeds_limit(prompts.preprocess())
        
@BaseReasoner.register(name='world_model')    
class WorldReasoner(BaseReasoner): 
    
    def __init__(self, 
                 generator: BaseLLMGenerator,
                 answer_prompt: BasePromptTemplate, 
                 question_prompt: BasePromptTemplate, 
                 llm_output_filter: filter_output_type
                 ) -> None:
        
        self.actor = Actor(generator)
        self.world_model = WorldModel(generator)
        self.answer_prompt = answer_prompt
        self.question_prompt = question_prompt
        self.llm_output_filter = llm_output_filter
    
    def generate_answer(self, 
                        idx: int, 
                        sample: dict[str, str], 
                        num_tries: int
                        ) -> Tuple[bool, bool, str, Panel | None]: 
        """ 
        Performs eval on one question from GSM8K and returns if successfully parsed, 
        and if llm was correct or not
        """
        question = sample['question']
        self.question_prompt.add('user', content=question)
        self.answer_prompt.add('user', content=question)
        message, panel = f'Answer Incorrect or failed to Generate for Question :(', None
        
        try: 
            for _ in range(num_tries): 
                sub_q = self.actor.act(self.question_prompt)
                self.question_prompt.add(**{'role': 'assistant', 'content': sub_q})
                self.answer_prompt.add(**{'role': 'user', 'content': sub_q})

                sub_a = self.world_model.step(self.answer_prompt)
                self.question_prompt.add('user', sub_a)
                self.answer_prompt.add('assistant', sub_a)
            
                if self.llm_output_filter(sub_a) == 'final_answer':
                    out, message = gsm_is_correct(idx, sub_a, sample)
                    return True, out, message, panel
            return True, False, message, panel # if run out of times, assume false
                
        except Exception as e: 
            return False, False, message, panel
        
    def reset_pass(self) -> None: 
        self.question_prompt.reset()
        self.answer_prompt.reset()
        
        
    def batch_generate_answer(self,
                            indices: list[int],
                            batched_samples: list[GSM8KProblem],
                            num_tries: int
                            ) -> Tuple[bool, 
                                        list[bool], 
                                        list[str], 
                                        list[Panel | None]
                                        ]:
                                
                                
        ...
        
    @classmethod
    def initialize(cls: Self, 
                   generator: BaseLLMGenerator, 
                   filter_output_func: Callable = filter_output_type
                   ) -> Self:
        
        question_prompt = GSMLlamaPromptTemplate('question', 1, 'question')
        answer_prompt = GSMLlamaPromptTemplate('answer', 1, 'answer')
        
        return cls(generator, 
                   question_prompt=question_prompt, 
                   answer_prompt=answer_prompt, 
                   llm_output_filter=filter_output_func)