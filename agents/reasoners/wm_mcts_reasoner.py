""" MCTS Reasoner that uses Two agents: world model agent and action agent to reason through a problem """

from __future__ import annotations
from typing import Callable, Any, Tuple

from agents.generators import BaseLLMGenerator
from agents.generators.vllm_generator import VLLMGenerator
from agents.reasoners.base_reasoner import BaseReasoner
from agents.prompts import BasePromptTemplate
from agents.gsm8k.utils import filter_output_type, gsm_is_correct

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