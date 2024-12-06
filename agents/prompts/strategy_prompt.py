"""
Class for constructing in Llama prompts for LLama-3 Vllm Chat Models 
"""
from __future__ import annotations
import json
import re
import os
import random
from typing import TypedDict, TypeVar, Optional, Any, Callable, Literal, Tuple, List, Self
from textwrap import dedent
import itertools
import pprint as pp
from enum import Enum
import copy
from dataclasses import dataclass

from agents.utils import BaseConfig
from agents.prompts.base_prompt_template import BasePromptTemplate
from agents.prompts.gsm_llama_prompts import BASE, QUESTION, ANSWER
from agents.gsm8k.utils import read_jsonl_dataset
# from agents.mcts.bigtree.bigtree_mcts_node import BTMCTSNode
# from agents.prompts.llama_prompt import GSMLlamaPromptTemplate
# from agents.mcts.bigtree import NodePath

class GSM8kPromptDict(TypedDict):
    """ Stores the Components for the Prompt """
    instruction: str
    interactive_examples: list[str]
    useful_examples: list[str]
    question_prefix: str
    subquestion_prefix: str
    overall_question_prefix: str
    answer_prefix: str


@dataclass
class PromptMessage:
    """ 
    PromptMessage container for role and content

    fields: 
    ======
    * role - Literal['user', 'assistant', 'system']
    * content - str
    """
    role: Literal['user', 'assistant', 'system']
    content: str


def parse_base(text: str) -> list[dict[str, str]]:
    chat_messages = []

    # Match blocks of text for system, user, and assistant roles
    pattern = r"(system|user|assistant):\n([\s\S]*?)(?=(?:system|user|assistant):|\Z)"
    matches = re.findall(pattern, text)

    # Iterate through matches and create chat message dicts
    for role, content in matches:
        # Strip extra whitespace and normalize content
        clean_content = content.strip()
        chat_messages.append({'role': role, 'content': clean_content})

    return chat_messages


def extract_question_idx(content: str) -> Optional[int]:
    """Extracts the question number from a given string."""
    match = re.search(r"Question (\d+):", content)
    if match:
        return int(match.group(1))
    return None


def format_prompt_messages(messages: List[PromptMessage]) -> str:
    """Formats a list of PromptMessage objects into the desired structured output."""
    output = []
    idx = None
    answer_guess_section = []

    for i, message in enumerate(messages):
        if message.role == 'user' and idx is None:
            # Extract question index from the first user message
            idx = extract_question_idx(message.content)

        if message.role == 'user':
            if i == 0:  # First user message is the math problem
                output.append(
                    f"user:\n** Math Problem **\nQuestion {idx}: {message.content.split(':', 1)[1].strip()}")
            else:  # Subsequent user messages are part of Answer Guess
                answer_guess_section.append(message.content)
        elif message.role == 'assistant':
            # Assistant content gets integrated into the answer guess
            answer_guess_section.append(message.content)

    # Combine all answer guesses into one structured section
    if answer_guess_section:
        output.append(f"user:\n** Answer Guess **\n" +
                      "\n".join(answer_guess_section))

    return "\n\n".join(output)


class GSMStrategyPromptTemplate(BasePromptTemplate):
    """Question-Answer Prompt Template for Llama."""

    base: str = """
system:
You are a brilliant agent that is good at creating strategies for solving math problems.     
Your goal is to come up with a high-level strategy for how to think about these kinds of problems. 
At each round of conversation, I will give you a math problem to solve, along with my best answer guess to the problem and the gold truth solution to the problem. 
Your goal is to reflect on why my answer is wrong, and provide a very high level strategy for how to improve on my problem solving skills.
First, return a reflection on what went wrong with the problem, being specific and thorough. Then, return a high-level strategy so that I can avoid this mistake in the future. 
While you can think critically about the problem at hand, the final strategy you return must be agnostic to the problem since I will use your strategy for other different math problems. Keep your strategy as one string.

Return your reflection on what went wrong with the following format: 
** Reflection **\n

Return your final strategy with the following format: 
** Strategy **\n
...

user: 
** Math Problem **
Question {idx}: Natalia sold clips to 48 of her friends in April, and then she sold half as many clips in May. How many clips did Natalia sell altogether in April and May?

user:
** Answer Guess **
Question {idx}.1: How many clips did Natalia sell in May?
Answer {idx}.1: Natalia sold 48 clips in April and half as many clips in May, so she sold 48 / 2 = 24 clips in May. The answer is 48.
Question {idx}.2: Now we can answer the question: How many clips did Natalia sell altogether in April and May?
Answer {idx}.2: Natalia sold 48 clips in April and 24 clips in May, so altogether she sold 48 + 24 = 72 clips. The answer is 72 clips.

** Gold Answer **
Answer {idx}.1: Natalia sold 48 clips in April and half as many clips in May, so she sold 48 / 2 = 24 clips in May. The answer is 48.
Answer {idx}.2: Natalia sold 48 clips in April and 24 clips in May, so altogether she sold 48 + 24 = 72 clips. The answer is #### 72.

** Result **
Answer Guess is WRONG.

assistant:
** Reflection ** 
The reason why your question was wrong was because your final answer was not output in the right format. The final answer should have been output as #### 72.

** Strategy **
You seem to be bad at outputing your answer in the right format. Make sure when you are giving your final answer, you put your answer in ####.

user: 
** Math Problem **
Question {idx}: Natalia sold clips to 48 of her friends in April, and then she sold half as many clips in May. How many clips did Natalia sell altogether in April and May?

user:
** Answer Guess **
Question {idx}.1: How many clips did Natalia sell in May?
Answer {idx}.1: Natalia sold 48 clips in April and half as many clips in May, so she sold 48 / 2 = 12 clips in May. The answer is 12.
Question {idx}.2: Now we can answer the question: How many clips did Natalia sell altogether in April and May?
Answer {idx}.2: Natalia sold 48 clips in April and 12 clips in May, so altogether she sold 48 + 12 = 60 clips. The answer is #### 60.

** Gold Answer **
Answer {idx}.1: Natalia sold 48 clips in April and half as many clips in May, so she sold 48 / 2 = 24 clips in May. The answer is 48.
Answer {idx}.2: Natalia sold 48 clips in April and 24 clips in May, so altogether she sold 48 + 24 = 72 clips. The answer is #### 72.

** Result **
Answer Guess is WRONG.

assistant:
** Reflection **
The error in your solution stems from a misunderstanding of the phrase "half as many clips." Instead of correctly calculating half of 48 as 24, you incorrectly concluded it was 12. This suggests an issue with accurately interpreting proportional relationships or operations described in the problem. Additionally, your final answer was formatted correctly, but it did not reflect the accurate calculations.

** Strategy **
Carefully read and interpret the problem to ensure you understand key relationships and operations. Translate descriptive phrases into explicit mathematical expressions and verify their correctness. Perform calculations systematically, double-checking intermediate results for consistency with the problem's context. Ensure your final answer is formatted as required and accurately reflects the correct calculations and problem conditions.
"""

    def __init__(self) -> None:
        """Initialize the GSMLlamaPromptTemplate."""
        fsl_prompt_base: List[dict[str, str]] = parse_base(self.base)
        # Initialize prompts and history
        self._base_prompt: List[PromptMessage] = [
            PromptMessage(**item) for item in fsl_prompt_base]
        self._prompt: List[PromptMessage] = copy.deepcopy(self._base_prompt)
        self._history: List[PromptMessage] = []

        self._prompt_kwargs = {}
        self._fsl_prompt_type = 'strategy'

    @property
    def prompt_kwargs(self) -> dict[str, Any]:
        """Returns the keyword arguments used to initialize the prompt."""
        return self._prompt_kwargs.copy()

    @property
    def prompt_type(self) -> str:
        """Returns the type of the prompt."""
        return self._fsl_prompt_type

    @property
    def history(self) -> List[PromptMessage]:
        """Returns a copy of the change history to prevent in-place modification."""
        return copy.deepcopy(self._history)

    def add(self, role: Literal['user', 'assistant', 'system'], content: str) -> None:
        """Add a new message to the prompt."""
        message = PromptMessage(role=role, content=content)
        self._prompt.append(message)
        self._history.append(message)

    def pop(self, indices: List[int] = [-1]) -> None:
        """Remove messages at specified indices from the prompt and adjust the history."""
        indices_set = {i if i >= 0 else len(self._prompt) + i for i in indices}
        indices_set = {i for i in indices_set if 0 <= i < len(self._prompt)}
        # Remove from prompt
        self._prompt = [msg for idx, msg in enumerate(
            self._prompt) if idx not in indices_set]
        # Adjust history
        self._history = [msg for msg in self._history if msg in self._prompt]

    def reset(self) -> None:
        """Reset the prompt to the base prompt and clear the history."""
        self._prompt = copy.deepcopy(self._base_prompt)
        self._history.clear()

    def preprocess(self) -> List[dict[str, str]]:
        """
        Preprocess the prompt messages for the model.
        Returns a list of dictionaries with 'role' and 'content'.
        """
        return [{'role': msg.role, 'content': msg.content} for msg in self._prompt]

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(prompt={pp.pformat(self._prompt)})"

    def __str__(self) -> str:
        return pp.pformat(self._prompt)

    def copy_history(self, prompt: 'GSMStrategyPromptTemplate') -> None:
        """
        Copy the history from another GSMLlamaPromptTemplate instance.
        If the prompt types differ, swap 'user' and 'assistant' roles in the copied history.
        """
        if not isinstance(prompt, GSMStrategyPromptTemplate):
            raise TypeError(
                "The provided prompt must be an instance of GSMStrategyPromptTemplate.")

        # Determine if role swapping is needed
        if self._fsl_prompt_type != prompt._fsl_prompt_type:
            # swapped_history = [
            #     PromptMessage(
            #         role='assistant' if msg.role == 'user' else 'user' if msg.role == 'assistant' else msg.role,
            #         content=msg.content
            #     )
            #     for msg in prompt.history
            # ]
            pass
        else:
            swapped_history = copy.deepcopy(prompt.history)

        # Update the current prompt and history
        self._history.extend(swapped_history)
        self._prompt.extend(swapped_history)

    def add_eval(self,
                 gold_trajectory: dict[str, str], # dict[] --> extract answer from 
                 mcts_prompt: BasePromptTemplate | GSMStrategyPromptTemplate, # the returned final mcts_prompt 
                 correct: bool
                 ) -> None:
        """ Adds the prompt from the leaf node of the optimal path """
        tf_map = {True: 'CORRECT', 
                  False: 'WRONG'}
        answer: str = gold_trajectory['answer'] # multi-line answer
        assistant_msg: str = f'\n\n** Gold Answer **\n{answer}'
        str_prompt_history: str = self.format_prompt_messages(mcts_prompt.history)
        single_traj: str = str_prompt_history + assistant_msg
        result = single_traj + f'\n\n** Result **:\nAnswer Guess is {tf_map.get(correct)}'
        self.add(role='user', content=result)
        
    def format_prompt_messages(self, messages: List[Any]) -> str:
        """Formats a list of messages into the desired structured output.
        
        The messages can be either PromptMessage objects or dicts with 'role' and 'content'.
        If it's PromptMessage, access them by message.role and message.content.
        If it's dict, access by message['role'] and message['content'].
        """
        output = []
        idx = None
        answer_guess_section = []

        def get_role_and_content(msg):
            if isinstance(msg, PromptMessage):
                return msg.role, msg.content
            elif isinstance(msg, dict):
                return msg.get('role', ''), msg.get('content', '')
            else:
                # If the message is neither a dict nor PromptMessage, handle gracefully
                # We'll default to user role and empty content.
                return 'user', str(msg)

        for i, message in enumerate(messages):
            role, content = get_role_and_content(message)

            # Extract question index only once from the first user message
            if role == 'user' and idx is None:
                idx = extract_question_idx(content)
                if idx is None:
                    # If we cannot find a question index, default to 1
                    idx = 1

            if role == 'user':
                if i == 0:
                    # First user message considered the main math problem
                    parts = content.split(':', 1)
                    if len(parts) > 1:
                        question_text = parts[1].strip()
                    else:
                        # If we can't split properly, just use entire content
                        question_text = content.strip()

                    output.append(
                        f"user:\n** Math Problem **\nQuestion {idx}: {question_text}")
                else:
                    # Subsequent user messages are part of answer guesses
                    answer_guess_section.append(content)
            elif role == 'assistant':
                # Assistant content also considered as part of the answer guess section
                answer_guess_section.append(content)
            else:
                # If an unexpected role appears, handle gracefully by ignoring or treating as user
                # Here we just ignore
                pass

        # Combine all answer guesses into one structured section
        if answer_guess_section:
            output.append(f"user:\n** Answer Guess **\n" + "\n".join(answer_guess_section))

        return "\n\n".join(output)




if __name__ == "__main__":
    path = '/lus/eagle/projects/FoundEpidem/bhsu/2024_research/agents/agents/data/gsm.jsonl'
    dataset = read_jsonl_dataset(path)[0]
    # Question 1.1: How many clips did Natalia sell in May?
    # Answer 1.1: Natalia sold 48 clips in April and half as many clips in May, so she sold 48 / 2 = 24 clips in May. The answer is 48.
    # Question 1.2: Now we can answer the question: How many clips did Natalia sell altogether in April and May?
    # Answer 1.2: Natalia sold 48 clips in April and 24 clips in May, so altogether she sold 48 + 24 = 72 clips. The answer is 72 clips.
    breakpoint()
    question_prompt = GSMLlamaPromptTemplate('question', 1, 'question')
    question_prompt.add(
        **{'role': 'user', 'content': 'Question 1: Natalia sold clips to 48 of her friends in April, and then she sold half as many clips in May. How many clips did Natalia sell altogether in April and May?'})
    question_prompt.add(
        **{'role': 'assistant', 'content': 'Question 1.1: How many clips did Natalia sell in May?'})
    question_prompt.add(
        **{'role': 'user', 'content': 'Answer 1.1: Natalia sold 48 clips in April and half as many clips in May, so she sold 48 / 2 = 24 clips in May. The answer is 48.'})
    question_prompt.add(
        **{'role': 'assistant', 'content': 'Question 1.2: Now we can answer the question: How many clips did Natalia sell altogether in April and May?'})
    question_prompt.add(
        **{'role': 'user', 'content': 'Answer 1.2: Natalia sold 48 clips in April and 24 clips in May, so altogether she sold 48 + 24 = 72 clips. The answer is 72 clips.'})
    breakpoint()
    
    out = format_prompt_messages(question_prompt.history)
    breakpoint()
    prompt = GSMStrategyPromptTemplate()
    # prompt.add_eval(mcts_prompt=prompt, gold_trajectory=)

    breakpoint()



"""
system:
You are a brilliant agent that is good at creating strategies for solving math problems.     
Your goal is to come up with a high-level strategy for how to think about these kinds of problems. 
At each round of conversation, I will give you a math problem to solve, along with my best answer guess to the problem and the gold truth solution to the problem. 
Your goal is to reflect on why my answer is wrong, and provide a very high level strategy for how to improve on my problem solving skills.
First, return a reflection on what went wrong with the problem, being specific and thorough. Then, return a high-level strategy so that I can avoid this mistake in the future. 
While you can think critically about the problem at hand, the final strategy you return must be agnostic to the problem since I will use your strategy for other different math problems. Keep your strategy as one string.

Return your reflection on what went wrong with the following format: 
** Reflection **\n

Return your final strategy with the following format: 
** Strategy **\n
...

user: 
** Math Problem **
Question {idx}: Natalia sold clips to 48 of her friends in April, and then she sold half as many clips in May. How many clips did Natalia sell altogether in April and May?

user:
** Answer Guess **
Question {idx}.1: How many clips did Natalia sell in May?
Answer {idx}.1: Natalia sold 48 clips in April and half as many clips in May, so she sold 48 / 2 = 24 clips in May. The answer is 48.
Question {idx}.2: Now we can answer the question: How many clips did Natalia sell altogether in April and May?
Answer {idx}.2: Natalia sold 48 clips in April and 24 clips in May, so altogether she sold 48 + 24 = 72 clips. The answer is 72 clips.

** Gold Answer **
Question {idx}.1: How many clips did Natalia sell in May?
Answer {idx}.1: Natalia sold 48 clips in April and half as many clips in May, so she sold 48 / 2 = 24 clips in May. The answer is 48.
Question {idx}.2: Now we can answer the question: How many clips did Natalia sell altogether in April and May?
Answer {idx}.2: Natalia sold 48 clips in April and 24 clips in May, so altogether she sold 48 + 24 = 72 clips. The answer is #### 72.

** Result **
Answer Guess is WRONG.

assistant:
** Reflection ** 
The reason why your question was wrong was because your final answer was not output in the right format. The final answer should have been output as #### 72.

** Strategy **
You seem to be bad at outputing your answer in the right format. Make sure when you are giving your final answer, you put your answer in ####.

user: 
** Math Problem **
Question {idx}: Natalia sold clips to 48 of her friends in April, and then she sold half as many clips in May. How many clips did Natalia sell altogether in April and May?

user:
** Answer Guess **
Question {idx}.1: How many clips did Natalia sell in May?
Answer {idx}.1: Natalia sold 48 clips in April and half as many clips in May, so she sold 48 / 2 = 12 clips in May. The answer is 12.
Question {idx}.2: Now we can answer the question: How many clips did Natalia sell altogether in April and May?
Answer {idx}.2: Natalia sold 48 clips in April and 12 clips in May, so altogether she sold 48 + 12 = 60 clips. The answer is #### 60.

** Gold Answer **
Question {idx}.1: How many clips did Natalia sell in May?
Answer {idx}.1: Natalia sold 48 clips in April and half as many clips in May, so she sold 48 / 2 = 24 clips in May. The answer is 48.
Question {idx}.2: Now we can answer the question: How many clips did Natalia sell altogether in April and May?
Answer {idx}.2: Natalia sold 48 clips in April and 24 clips in May, so altogether she sold 48 + 24 = 72 clips. The answer is #### 72.

** Result **
Answer Guess is WRONG.

assistant:
** Reflection **
The error in your solution stems from a misunderstanding of the phrase "half as many clips." Instead of correctly calculating half of 48 as 24, you incorrectly concluded it was 12. This suggests an issue with accurately interpreting proportional relationships or operations described in the problem. Additionally, your final answer was formatted correctly, but it did not reflect the accurate calculations.

** Strategy **
Carefully read and interpret the problem to ensure you understand key relationships and operations. Translate descriptive phrases into explicit mathematical expressions and verify their correctness. Perform calculations systematically, double-checking intermediate results for consistency with the problem's context. Ensure your final answer is formatted as required and accurately reflects the correct calculations and problem conditions.
"""