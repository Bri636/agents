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

from agents.gsm8k.types import GSM8KProblem
from agents.mcts import T
from agents.mcts.bigtree.bigtree_llm_mcts import MCTS
from agents.mcts.bigtree.batch_bigtree_llm_mcts import BatchMCTS
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
        
    def batch_step_logprobs(self, answer_prompts: list[BasePromptTemplate]) -> list[dict]:
        """ Batch generates the next state with log probabilities """
        assert isinstance(self.generator, VLLMGenerator), f"""
        LogProbs only supported with VLLM for now...
        """
        answer_inputs = [answer_prompt.preprocess() for answer_prompt in answer_prompts]
        sub_answers = self.generator.generate_with_logprobs(answer_inputs)
        return [{'text': ans['text'][0], 'log_probs': ans['log_probs']} for ans in sub_answers]

    def prompt_exceeds_limit(self, prompts: BasePromptTemplate):
        return self.generator.prompt_exceeds_limit(prompts.preprocess())


class Actor:
    def __init__(self, generator: BaseLLMGenerator) -> None:

        self.generator = generator

    def act(self, question_prompt: BasePromptTemplate) -> str:
        """ Returns the next sub_question to ask"""
        sub_question = self.generator.generate(question_prompt.preprocess())[0]

        return sub_question
    
    def batch_act(self, question_prompts: list[BasePromptTemplate]) -> list[str]:
        """ Batch returns the next sub_question to ask """
        question_inputs: list[list[dict]] = [question_prompt.preprocess()
                                             for question_prompt in question_prompts]  # each sub-list is a chat buffer
        sub_questions = self.generator.generate(question_inputs)
        return sub_questions

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
        mcts = MCTS(question_prompt_base=self.question_prompt, answer_prompt_base=self.answer_prompt)

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

    def batch_generate_answer(self,
                              indices: list[int],
                              samples: list[dict[str, str]],
                              num_tries: int,
                              num_children: int = 3
                              ) -> Tuple[bool, list[bool], list[str], list[Panel | None]]:
        """
        Attempts to generate answers for a batch of samples.
        Returns:
            - A flag indicating if the batch was processed successfully.
            - A list of booleans indicating if each answer was correct.
            - A list of messages for each sample.
            - A list of panels (visualizations) for each sample.
        """
        batch_size = len(samples)
        generated = [False] * batch_size # assume false
        correct = [False] * batch_size
        messages = ['Answer incorrect or failed to generate for question.'] * batch_size
        panels = [None] * batch_size
        
        roots = []
        for idx in range(batch_size): 
            # prime the prompts with the problem
            problem: GSM8KProblem = samples[idx]['question']
            question_prompt: BasePromptTemplate = copy.deepcopy(self.question_prompt)
            answer_prompt: BasePromptTemplate = copy.deepcopy(self.answer_prompt)
            question_prompt.add('user', content=problem)
            answer_prompt.add('user', content=problem)
            
            root = BTMCTSNode(
                state=question_prompt,
                action=None,
                reward=None,
                parent=None,
                is_terminal=False
            )
            roots.append(root)

        # note - we deepcopy to prevent over-writing question_prompt
        mcts = BatchMCTS(question_prompt_base=copy.deepcopy(question_prompt), 
                         answer_prompt_base=copy.deepcopy(answer_prompt))

        answer, optimal_path, panels = mcts.batch_guess_answer(roots, 
                                                               self.actor, 
                                                               self.world_model, 
                                                               num_children=num_children, 
                                                               samples=samples, 
                                                               sample_indices=indices, 
                                                               max_tries=num_tries
                                                               )
        breakpoint()
        # # Initialize MCTS instances and roots for each sample
        # roots = []
        # for idx in range(batch_size):
        #     question = samples[idx]['question']
        #     question_prompt = copy.deepcopy(self.question_prompt)
        #     answer_prompt = copy.deepcopy(self.answer_prompt)
        #     question_prompt.add('user', content=question)
        #     answer_prompt.add('user', content=question)

        #     root = BTMCTSNode(
        #         state=question_prompt,
        #         action=None,
        #         reward=None,
        #         parent=None,
        #         is_terminal=False
        #     )
        #     roots.append(root)

        # mcts = BatchMCTS(
        #         question_prompt_base=question_prompt,
        #         answer_prompt_base=answer_prompt
        #     )
        # ############### mine 
        # # NOTE - expansion and simulation must take in different questions for them to be batched
        # # aka one mcts interface per batch
        # answer, optimal_path, panels = mcts.batch_guess_answer(roots, 
        #                                                        self.actor, 
        #                                                        self.world_model, 
        #                                                        num_children=3, 
        #                                                        samples=samples, 
        #                                                        sample_indices=indices, 
        #                                                        max_tries=5
        #                                                        )
        
        # ########### gpt ##############

        # # Process MCTS instances
        # for idx in range(batch_size):
        #     mcts: BatchMCTS = mcts_instances[idx]
        #     root: BTMCTSNode = roots[idx]
        #     try:
        #         answer, optimal_path, panel = mcts.batch_guess_answer(
        #             root=root,
        #             actor=self.actor,
        #             world_model=self.world_model,
        #             sample=samples[idx],
        #             sample_idx=indices[idx],
        #             max_tries=num_tries,
        #             num_children=num_children,
        #             verbose=False  # Set to True if needed
        #         )
        #         if self.llm_output_filter(answer) == 'final_answer':
        #             correct[idx], message = gsm_is_correct(indices[idx], answer, samples[idx])
        #             messages[idx] = message
        #         generated[idx] = True
        #         panels[idx] = panel
        #     except Exception as e:
        #         messages[idx] = f"Error processing sample {indices[idx]}: {e}"

        # overall_success = all(generated)
        # return overall_success, correct, messages, panels

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
