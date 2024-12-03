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


# from agents.mcts.bigtree.bigtree_llm_mcts import MCTS
# from agents.mcts.bigtree.batch_bigtree_llm_mcts import BatchMCTS
# from agents.mcts.bigtree.bigtree_mcts_node import BTMCTSNode
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
        sub_answer = self.generator.generate_with_logprobs(
            answer_prompt.preprocess())
        return {'text': sub_answer['text'][0],
                'log_probs': sub_answer['log_probs'],
                }
        
    def batch_step_logprobs(self, answer_prompts: list[BasePromptTemplate]) -> list[dict[str, str | float]]:
        """ Batch generates the next state with log probabilities """
        assert isinstance(self.generator, VLLMGenerator), f"""
        LogProbs only supported with VLLM for now...
        """
        answer_inputs = [answer_prompt.preprocess() for answer_prompt in answer_prompts]
        sub_answers = self.generator.generate_with_logprobs(answer_inputs)
        
        texts = [sub_answer for sub_answer in sub_answers['text']]
        log_probs = [log_prob for log_prob in sub_answers['log_probs']]

        return [{'text': text, 'log_probs': log_prob} 
                for text, log_prob in zip(texts, log_probs)]

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
                              sample_indices: list[int],
                              samples: list[dict[str, str]],
                              num_tries: int,
                              num_children: int = 3, 
                              verbose: bool = True
                              ) -> Tuple[bool, list[bool], list[str], list[Panel | None]]:
        """
        Attempts to generate answers for a batch of samples.
        Returns:
            - A flag indicating if the batch was processed successfully.
            - A list of booleans indicating if each answer was correct.
            - A list of messages for each sample.
            - A list of panels (visualizations) for each sample.
        """
        from agents.mcts.bigtree.batch_bigtree_llm_mcts import BatchMCTS
        from agents.mcts.bigtree.bigtree_mcts_node import BTMCTSNode
        
        batch_size = len(samples)
        corrects = [False] * batch_size
        messages = ['Answer incorrect or failed to generate for question.'] * batch_size
        panels = [None] * batch_size

        roots: list[BTMCTSNode] = []
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

        try: 
            answers, optimal_paths, panels = mcts.batch_guess_answer(roots, 
                                                                self.actor, 
                                                                self.world_model, 
                                                                num_children=num_children, 
                                                                samples=samples, 
                                                                sample_indices=sample_indices, 
                                                                max_tries=num_tries
                                                                )
            for idx, (sample_idx, sample, answer) in enumerate(zip(sample_indices, samples, answers)): 
                filtered_answer = self.llm_output_filter(answer)
                if filtered_answer == 'final_answer':
                        correct, message = gsm_is_correct(sample_idx, answer, sample)
                        # update containers
                        corrects[idx] = correct
                        messages[idx] = message
            return True, corrects, messages, panels
            
        except Exception as e:
            messages = [f'Failed to generate due to this error: {e}, dropping batch...\n'] * batch_size
            breakpoint()
            return False, [False] * batch_size, messages, panels # else, False and drop batch 

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
