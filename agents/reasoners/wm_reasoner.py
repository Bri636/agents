""" Reasoner that uses Two agents: world model agent and action agent to reason through a problem """

from __future__ import annotations
from typing import Callable, Any, Tuple, Optional, Self
from rich.panel import Panel
import copy

from agents.prompts.llama_prompt import GSMLlamaPromptTemplate
from agents.generators import BaseLLMGenerator
from agents.generators.vllm_generator import VLLMGenerator
from agents.reasoners.base_reasoner import BaseReasoner
from agents.prompts import BasePromptTemplate
from agents.gsm8k.utils import filter_output_type, gsm_is_correct
from agents.gsm8k.types import GSM8KProblem

PromptItem = list[Tuple[BasePromptTemplate, BasePromptTemplate], bool]
""" Item in Batch - [QUESTION_PROMPT, ANSWER_PROMPT, IF_CORRECT] """


class WorldModel:
    def __init__(self, generator: BaseLLMGenerator) -> None:

        self.generator = generator

    def step(self, answer_prompt: BasePromptTemplate) -> str:
        """ Generates the next state given the current action or state """
        sub_answer = self.generator.generate(answer_prompt.preprocess())[0]

        return sub_answer

    def batch_step(self, answer_prompts: list[BasePromptTemplate]) -> list[str]:
        """ Batch Generates the next state given the current action or state """
        answer_inputs = [answer_prompt.preprocess()
                         for answer_prompt in answer_prompts]
        sub_answers = self.generator.generate(answer_inputs)
        return sub_answers

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
                self.question_prompt.add(
                    **{'role': 'assistant', 'content': sub_q})
                self.answer_prompt.add(**{'role': 'user', 'content': sub_q})

                sub_a = self.world_model.step(self.answer_prompt)
                self.question_prompt.add('user', sub_a)
                self.answer_prompt.add('assistant', sub_a)

                if self.llm_output_filter(sub_a) == 'final_answer':
                    out, message = gsm_is_correct(idx, sub_a, sample)
                    return True, out, message, panel
            return True, False, message, panel  # if run out of times, assume false

        except Exception as e:
            return False, False, message, panel

    def reset_pass(self) -> None:
        self.question_prompt.reset()
        self.answer_prompt.reset()

    def batch_reset(self, prompts: list[PromptItem]) -> None:
        """ Resets the history of a batch of prompts to base """
        for prompt in prompts: 
            prompt[0].reset()
            prompt[1].reset()

    def batch_step(self, filtered_prompts: list[PromptItem]) -> dict[str, Tuple[list[BasePromptTemplate] | list[str],
                                                                               list[BasePromptTemplate] | list[str]]]:
        """ 
        Adds a single chat interaction for actor-world_model based on a list of question and answer prompts

        * NOTE - performs in-place ops on the prompts so beware
        """
        # grab question and answer prompts from filtered_prompts
        question_prompts = [prompt[0] for prompt in filtered_prompts]
        answer_prompts = [prompt[1] for prompt in filtered_prompts]
        
        sub_questions = self.actor.batch_act(question_prompts)
        # Add generated sub-questions to respective prompts
        for question_prompt, answer_prompt, sub_q in zip(question_prompts, answer_prompts, sub_questions):
            question_prompt.add(**{'role': "assistant", 'content': sub_q})
            answer_prompt.add(**{'role': "user", 'content': sub_q})

        # Batch step to get sub-answers
        sub_answers = self.world_model.batch_step(answer_prompts)
        # Add generated sub-answers to respective prompts
        for question_prompt, answer_prompt, sub_a in zip(question_prompts, answer_prompts, sub_answers):
            question_prompt.add(**{'role': "user", 'content': sub_a})
            answer_prompt.add(**{'role': "assistant", 'content': sub_a})

        return {
            'responses': (sub_questions, sub_answers),
            'prompts': (question_prompts, answer_prompts)
        }

    def batch_generate_answer(self,
                              indices: list[int],
                              batched_samples: list[GSM8KProblem],
                              num_tries: int
                              ) -> Tuple[bool,
                                         list[bool],
                                         list[str],
                                         list[Panel | None]
                                         ]:
        """ Batched generation for WM Strategy """
        batch_size = len(batched_samples)
        problems: list[str] = [sample['question']
                                for sample in batched_samples]
        question_prompts: list[BasePromptTemplate] = [copy.deepcopy(self.question_prompt)
                                                      for _ in range(batch_size)]
        answer_prompts: list[BasePromptTemplate] = [copy.deepcopy(self.answer_prompt)
                                                    for _ in range(batch_size)]
        # load all the problems to their prompts
        for problem, question_prompt, answer_prompt in zip(problems,
                                                            question_prompts,
                                                            answer_prompts):
            question_prompt.add(**{'role': 'user', 'content': problem})
            answer_prompt.add(**{'role': 'user', 'content': problem})

        corrects = [False] * batch_size
        messages, panels = [
            f'Answer Incorrect or failed to Generate for Question :('] * batch_size, [None] * batch_size

        batch_prompts: list[PromptItem] = [[(question_prompt, answer_prompt), False] # bool for it problem is correct
                                          for question_prompt, answer_prompt
                                          in zip(question_prompts, answer_prompts)]

        try:
            for _ in range(num_tries):
                # filter out for prompts we still need to do - aka True
                filtered_prompts: list[PromptItem] = list(
                    filter(lambda item: not item[-1], batch_prompts))
                # batch output
                batch_out = self.batch_step(filtered_prompts)
                sub_questions, sub_answers = batch_out['responses'] # list[str] responses
                question_prompt, answer_prompt = batch_out['prompts'] # prompts with chat history updated
                # evaluate the filtered answers
                filtered_answers: list[str] = [self.llm_output_filter(sub_answer)
                                               for sub_answer in sub_answers]
                for idx, (filtered_answer, prompt, sample, sample_idx) in enumerate(
                    zip(filtered_answers, batch_prompts, batched_samples, indices)
                ):
                    if filtered_answer == 'final_answer':
                        correct, message = gsm_is_correct(
                            sample_idx, answer_prompt[idx], sample)
                        batch_prompts[idx][-1] = correct  # mark batch item as correct
                        # set message to correct message
                        messages[idx] = message
                        # set sample idx to correct result
                        corrects[idx] = correct
                        self.batch_reset(batch_prompts)
                        
                # if all of them were correct, leave for loop early 
                if all([prompt[-1] for prompt in batch_prompts]): 
                    break
            
            return True, corrects, messages, panels  # else, return best guess

        except Exception as e:
            messages = [f'Failed to generate due to this error: {e}\n'] * batch_size
            return False, [False] * batch_size, messages, panels # else, False and drop batch 

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
