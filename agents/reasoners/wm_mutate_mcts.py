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
from agents.mcts.bigtree.batch_bigtree_llm_mcts import BatchMCTS
from agents.mcts.bigtree.bigtree_mcts_node import BTMCTSNode
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
    
def extract_strategy(input_list: list[str]) -> list[str]:
    parsed_list = []
    for item in input_list:
        # Split the string by "** Strategy **"
        parts = item.split("** Strategy **")
        if len(parts) > 1:  # Check if "** Strategy **" exists in the string
            # Strip leading/trailing whitespace from the strategy part and append it to the result
            parsed_list.append(parts[1].strip())
    return parsed_list
    
class Mutator: 
    def __init__(self, generator: BaseLLMGenerator, parser: Callable = extract_strategy) -> None:

        self.generator = generator
        self.parser = parser

    def act(self, strategy_prompt: BasePromptTemplate) -> str:
        """ Returns the next sub_question to ask"""
        sub_question = self.generator.generate(strategy_prompt.preprocess())[0]

        return sub_question
    
    def batch_mutate(self, strategy_prompts: list[BasePromptTemplate]) -> list[str]:
        """ Batch returns the next sub_question to ask """
        strategy_inputs: list[list[dict]] = [strategy_prompt.preprocess()
                                             for strategy_prompt in strategy_prompts]  # each sub-list is a chat buffer
        strategy_raws = self.generator.generate(strategy_inputs)
        strategies = self.parser(strategy_raws)
        return strategies

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


@BaseReasoner.register(name='mutate_mcts_world_model')
class MutateMCTSWorldReasoner(BaseReasoner):

    def __init__(self,
                 generator: BaseLLMGenerator,
                 answer_prompt: BasePromptTemplate,
                 question_prompt: BasePromptTemplate,
                 llm_output_filter: filter_output_type,
                 **kwargs
                 ) -> None:

        self.actor = Actor(generator)
        self.world_model = WorldModel(generator)
        self.mutator = Mutator(generator)
        self.answer_prompt = answer_prompt
        self.question_prompt = question_prompt
        self.llm_output_filter = llm_output_filter
        
        self.strategies: dict[int, str] = {}

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
        from agents.mcts.bigtree.batch_bigtree_llm_mcts import BatchMCTS
        from agents.mcts.bigtree.bigtree_mcts_node import BTMCTSNode
        from agents.prompts.strategy_prompt import GSMStrategyPromptTemplate

        batch_size = len(samples)
        corrects = [False] * batch_size
        messages = ['Answer incorrect or failed to generate for question.'] * batch_size
        panels = [None] * batch_size

        roots: list[BTMCTSNode] = []
        for idx in range(batch_size):
            problem: GSM8KProblem = samples[idx]['question']
            question_prompt: BasePromptTemplate = copy.deepcopy(self.question_prompt)
            answer_prompt: BasePromptTemplate = copy.deepcopy(self.answer_prompt)
            question_prompt.add('user', content=problem)
            answer_prompt.add('user', content=problem)
            
            # If we have previously recorded strategies, apply them
            for idx, strategy in self.strategies.items():
                question_prompt.inject_strategy(strategy)
                answer_prompt.inject_strategy(strategy)

            root = BTMCTSNode(
                state=question_prompt,
                action=None,
                reward=None,
                parent=None,
                is_terminal=False
            )
            roots.append(root)

        # We deepcopy to avoid overwriting the originals
        mcts = BatchMCTS(question_prompt_base=copy.deepcopy(question_prompt),
                        answer_prompt_base=copy.deepcopy(answer_prompt))

        try:
            answers, optimal_paths, panels = mcts.batch_guess_answer(
                roots,
                self.actor,
                self.world_model,
                num_children=num_children,
                samples=samples,
                sample_indices=sample_indices,
                max_tries=num_tries
            )

            strategy_prompts: list[Tuple[int, GSMStrategyPromptTemplate]] = []

            for idx, (sample_idx, sample, answer, optimal_path) in enumerate(
                zip(sample_indices, samples, answers, optimal_paths)
            ):
                filtered_answer = self.llm_output_filter(answer)
                # Check if optimal_path is empty
                if not optimal_path:
                    # Handle the no-path scenario: 
                    # This could mean that MCTS failed to produce any expansions
                    # You might want to log this or simply continue to the next sample.
                    messages[idx] = "No optimal path found. MCTS failed to produce a solution."
                    continue

                if filtered_answer == 'final_answer':
                    correct, message = gsm_is_correct(sample_idx, answer, sample)
                    corrects[idx] = correct
                    messages[idx] = message

                    if not correct:
                        breakpoint()
                        # Now it's safe to access optimal_path[-1]
                        strategy_prompt: GSMStrategyPromptTemplate = GSMStrategyPromptTemplate()
                        strategy_prompt.add_eval(sample, optimal_path[-1].state, correct)
                        strategy_prompts.append((idx, strategy_prompt))
                # If we have prompts that need strategies
                if len(strategy_prompts) > 0:
                    # Extract the templates
                    templates = [p[1] for p in strategy_prompts]
                    # Mutate them to get strategies
                    strategies = self.mutator.batch_mutate(templates)
                    # Assign each returned strategy to the corresponding index
                    for (idx, _), strategy in zip(strategy_prompts, strategies):
                        self.strategies[idx] = strategy
            return True, corrects, messages, panels

        except Exception as e:
            messages = [f'Failed to generate due to this error: {e}, dropping batch...\n'] * batch_size
            return False, [False] * batch_size, messages, panels

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
