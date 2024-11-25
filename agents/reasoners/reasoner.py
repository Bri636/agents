""" Container class that uses two question-answer agents """

from __future__ import annotations
from typing import Callable, Tuple
from rich.panel import Panel
import copy

from agents.reasoners.base_reasoner import BaseReasoner
from agents.gsm8k.utils import filter_output_type, gsm_is_correct
from agents.generators import BaseLLMGenerator
from agents.prompts import BasePromptTemplate
from agents.gsm8k import GSM8KProblem
from agents.prompts.standard_prompt import StandardGSMPromptTemplate
# import types
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

        self.batch_prompts = ()

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

                llm_output = self.generator.generate(
                    self.prompt.preprocess())[0]

                filtered_output = self.llm_output_filter(llm_output)
                if filtered_output == 'final_answer':
                    self.prompt.reset()
                    correct, message = gsm_is_correct(idx, llm_output, sample)
                    return True, correct, message, panel

                elif filtered_output == '[invalid]':
                    pass

            return True, False, message, panel  # else, return best guess

        except Exception as e:
            return False, False, message, panel

    def reset_pass(self) -> None:
        self.prompt.reset()

    def batch_generate_answer(self,
                              indices: list[int],
                              batched_samples: list[GSM8KProblem],
                              num_tries: int
                              ) -> Tuple[bool, 
                                         list[bool], 
                                         list[str], 
                                         list[Panel | None]
                                         ]:
        """
        Performs batch eval on questions from GSM8K and returns if successfully parsed, 
        and if llm was correct or not 
        """
        batch_size = len(batched_samples)
        questions: list[str] = [sample['question'] 
                                for sample in batched_samples]
        prompts: list[BasePromptTemplate] = [copy.deepcopy(self.prompt)
                                             for _ in range(batch_size)]
        for question, prompt in zip(questions, prompts):
            prompt.add(**{'role': 'user', 'content': question})
            
        corrects = [False] * batch_size
        messages, panels = [
            f'Answer Incorrect or failed to Generate for Question :('] * batch_size, [None] * batch_size
        # each sublist is one chat prompt
        inputs: list[list[list[dict], bool]] = [[prompt.preprocess(), False]  # True if unfinished
                                                    for prompt in prompts]
        try:
            for _ in range(num_tries):

                filtered_inputs: list[Tuple[list[dict], bool]] = list(filter(lambda input: not input[1], inputs)) # filter for unfinished problems
                prompt_inputs: list[list[dict]] = [filtered_input[0] for filtered_input in filtered_inputs]
                llm_outputs: list[str] = self.generator.generate(prompt_inputs)
                filtered_outputs: list[str] = [self.llm_output_filter(llm_output)
                                               for llm_output in llm_outputs]
                for idx, (filtered_output, prompt, sample, sample_idx) in enumerate(
                    zip(filtered_outputs, prompts, batched_samples, indices)
                ):
                    if filtered_output == 'final_answer': 
                        correct, message = gsm_is_correct(sample_idx, llm_outputs[idx], sample)
                        inputs[idx][-1] = correct # set inputs idx to result 
                        messages[idx] = message # set message to correct message
                        corrects[idx] = correct # set sample idx to correct result
                        prompt.reset()
                # if all of them were correct, leave for loop early 
                if all([input[-1] for input in inputs]): 
                    break
            # return batch of results
            return True, corrects, messages, panels  # else, return best guess

        except Exception as e:
            messages = [f'Failed to generate due to this error: {e}\n'] * batch_size
            return False, [False] * batch_size, messages, panels # else, False and drop batch 

    @classmethod
    def initialize(cls: T,
                   generator: BaseLLMGenerator,
                   filter_output_func: Callable = filter_output_type
                   ) -> T:
        prompt = StandardGSMPromptTemplate()
        return cls(generator, prompt, filter_output_func)
