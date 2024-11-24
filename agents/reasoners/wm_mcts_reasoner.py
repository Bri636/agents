""" MCTS Reasoner that uses Two agents: world model agent and action agent to reason through a problem """

from __future__ import annotations
from typing import Callable, Any, Tuple, Self

from rich.panel import Panel
import copy 

from agents.generators import BaseLLMGenerator
from agents.generators.vllm_generator import VLLMGenerator
from agents.reasoners.base_reasoner import BaseReasoner
from agents.prompts import BasePromptTemplate
from agents.prompts.llama_prompt import GSMLlamaPromptTemplate
from agents.gsm8k.utils import filter_output_type, gsm_is_correct

from agents.mcts import T
from agents.mcts.bigtree.bigtree_llm_mcts import MCTS
from agents.mcts.bigtree.bigtree_mcts_node import BTMCTSNode

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
        sub_answer = self.generator.generate_with_logprobs(
            answer_prompt.preprocess())
        return {'text': sub_answer['text'][0],
                'log_probs': sub_answer['log_probs'],
                }

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
        sub_question = self.generator.generate_with_logprobs(
            question_prompt.preprocess())
        return {'text': sub_question['text'][0],
                'token_seq': sub_question['token_seq'],
                'log_probs': sub_question['log_probs'],
                }

    def prompt_exceeds_limit(self, prompts: BasePromptTemplate):
        return self.generator.prompt_exceeds_limit(prompts.preprocess())

@BaseReasoner.register(name='mcts_world_model')
class MCTSWorldReasoner(BaseReasoner):

    def __init__(self,
                 generator: BaseLLMGenerator,
                 answer_prompt: BasePromptTemplate,
                 question_prompt: BasePromptTemplate,
                 llm_output_filter: filter_output_type,
                 **kwargs
                 ) -> None:

        self.actor = Actor(generator)
        self.world_model = WorldModel(generator)
        self.answer_prompt = answer_prompt
        self.question_prompt = question_prompt
        self.llm_output_filter = llm_output_filter

    def generate_answer(self, 
                        idx: int, 
                        sample: dict[str, str], 
                        num_tries: int, 
                        num_children: int = 3
                        ) -> Tuple[bool, bool, str, Panel | None]:
        """ 
        Attempts to generate an answer for a sample question; it will return - 
        Tuple[if successfully generated, and if answer was correct]
        """
        question = sample['question']
        mcts = MCTS(question_prompt_base=self.question_prompt,
                    answer_prompt_base=self.answer_prompt,
                    )

        self.question_prompt.add('user', content=question)
        self.answer_prompt.add('user', content=question)
        # if answer was generated, and if answer was correct or not
        generated, correct = False, False
        message, panel = f'Answer Incorrect or failed to Generate for Question :(', None
        
        root = BTMCTSNode(state=self.question_prompt,  # state is original question
                          action=None,
                          reward=None,
                          parent=None,
                          is_terminal=False
                          )

        try:
            answer, optimal_path, panel = mcts.guess_answer(root=root,
                                                     actor=self.actor,
                                                     world_model=self.world_model,
                                                     sample=sample,
                                                     sample_idx=idx,
                                                     max_tries=num_tries,
                                                     num_children=num_children
                                                     )
            if self.llm_output_filter(answer) == 'final_answer':
                correct, message = gsm_is_correct(idx, answer, sample)

            generated = True
            return generated, correct, message, panel

        except Exception as e:
            return generated, correct, message, panel

    @classmethod
    def initialize(cls: Self, 
                   generator: BaseLLMGenerator, 
                   filter_output_func: Callable = filter_output_type
                   ) -> Self:
        
        question_prompt = GSMLlamaPromptTemplate('question', 1, 'question')
        answer_prompt = GSMLlamaPromptTemplate('answer', 1, 'answer')
        
        return cls(generator, 
                   answer_prompt=answer_prompt, 
                   question_prompt=question_prompt, 
                   llm_output_filter=filter_output_func)