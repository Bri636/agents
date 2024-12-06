# """
# Class for constructing in Llama prompts for LLama-3 Vllm Chat Models 
# """
# from __future__ import annotations
# import json
# import re
# import os
# import random
# from typing import TypedDict, TypeVar, Optional, Any, Callable, Literal, Tuple, List, Self
# from textwrap import dedent
# import itertools
# import pprint as pp
# from enum import Enum
# import copy
# from dataclasses import dataclass

# from agents.utils import BaseConfig
# from agents.prompts.base_prompt_template import BasePromptTemplate
# from agents.prompts.gsm_llama_prompts import BASE, QUESTION, ANSWER
# from agents.gsm8k.utils import read_jsonl_dataset
# from agents.mcts.bigtree.bigtree_mcts_node import BTMCTSNode
# from agents.prompts.llama_prompt import GSMLlamaPromptTemplate
# from agents.mcts.bigtree import NodePath

# class GSM8kPromptDict(TypedDict):
#     """ Stores the Components for the Prompt """
#     instruction: str
#     interactive_examples: list[str]
#     useful_examples: list[str]
#     question_prefix: str
#     subquestion_prefix: str
#     overall_question_prefix: str
#     answer_prefix: str


# @dataclass
# class PromptMessage:
#     """ 
#     PromptMessage container for role and content

#     fields: 
#     ======
#     * role - Literal['user', 'assistant', 'system']
#     * content - str
#     """
#     role: Literal['user', 'assistant', 'system']
#     content: str


# def parse_base(text: str) -> list[dict[str, str]]:
#     chat_messages = []

#     # Match blocks of text for system, user, and assistant roles
#     pattern = r"(system|user|assistant):\n([\s\S]*?)(?=(?:system|user|assistant):|\Z)"
#     matches = re.findall(pattern, text)

#     # Iterate through matches and create chat message dicts
#     for role, content in matches:
#         # Strip extra whitespace and normalize content
#         clean_content = content.strip()
#         chat_messages.append({'role': role, 'content': clean_content})

#     return chat_messages


# def extract_question_idx(content: str) -> Optional[int]:
#     """Extracts the question number from a given string."""
#     match = re.search(r"Question (\d+):", content)
#     if match:
#         return int(match.group(1))
#     return None


# def format_prompt_messages(messages: List[PromptMessage]) -> str:
#     """Formats a list of PromptMessage objects into the desired structured output."""
#     output = []
#     idx = None
#     answer_guess_section = []

#     for i, message in enumerate(messages):
#         if message.role == 'user' and idx is None:
#             # Extract question index from the first user message
#             idx = extract_question_idx(message.content)

#         if message.role == 'user':
#             if i == 0:  # First user message is the math problem
#                 output.append(
#                     f"user:\n** Math Problem **\nQuestion {idx}: {message.content.split(':', 1)[1].strip()}")
#             else:  # Subsequent user messages are part of Answer Guess
#                 answer_guess_section.append(message.content)
#         elif message.role == 'assistant':
#             # Assistant content gets integrated into the answer guess
#             answer_guess_section.append(message.content)

#     # Combine all answer guesses into one structured section
#     if answer_guess_section:
#         output.append(f"user:\n** Answer Guess **\n" +
#                       "\n".join(answer_guess_section))

#     return "\n\n".join(output)


# class GSMQuestionPromptTemplate(BasePromptTemplate):
#     """Question-Answer Prompt Template for Llama."""

#     base: str = """
# system:
# You are a brilliant agent that is good at coming up with sub-questions that correspond to a list of sub-answers.
# At each round I will give you a list of sub-answers and the final answer for a math problem. 
# Your goal is to come up with a list of sub-questions that correspond with the sub-answers. 
# The number of sub-questions you write MUST match the number of sub-answers. 

# You MUST return your sub-questions in the following format:
# #### Question 1: Your sub-question corresponding to sub-answer 1 for the overall problem 
# #### Question 2: Your sub-question corresponding to sub-answer 2 for the overall problem 
# ...

# user: 
# "Natalia sold clips to 48 of her friends in April, and then she sold half as many clips in May. How many clips did Natalia sell altogether in April and May?"
# Natalia sold 48 clips in April and half as many clips in May, so she sold 48 / 2 = 24 clips in May. The answer is 48.
# Natalia sold 48 clips in April and 24 clips in May, so altogether she sold 48 + 24 = 72 clips. The answer is #### 72.

# assistant: 
# #### Question 1: How many clips did Natalia sell in May?
# #### Question 2: Now we can answer the question: How many clips did Natalia sell altogether in April and May?

# """

#     def __init__(self) -> None:
#         """Initialize the GSMLlamaPromptTemplate."""
#         fsl_prompt_base: List[dict[str, str]] = parse_base(self.base)
#         # Initialize prompts and history
#         self._base_prompt: List[PromptMessage] = [
#             PromptMessage(**item) for item in fsl_prompt_base]
#         self._prompt: List[PromptMessage] = copy.deepcopy(self._base_prompt)
#         self._history: List[PromptMessage] = []

#         self._prompt_kwargs = {}
#         self._fsl_prompt_type = 'strategy'

#     @property
#     def prompt_kwargs(self) -> dict[str, Any]:
#         """Returns the keyword arguments used to initialize the prompt."""
#         return self._prompt_kwargs.copy()

#     @property
#     def prompt_type(self) -> str:
#         """Returns the type of the prompt."""
#         return self._fsl_prompt_type

#     @property
#     def history(self) -> List[PromptMessage]:
#         """Returns a copy of the change history to prevent in-place modification."""
#         return copy.deepcopy(self._history)

#     def add(self, role: Literal['user', 'assistant', 'system'], content: str) -> None:
#         """Add a new message to the prompt."""
#         message = PromptMessage(role=role, content=content)
#         self._prompt.append(message)
#         self._history.append(message)

#     def pop(self, indices: List[int] = [-1]) -> None:
#         """Remove messages at specified indices from the prompt and adjust the history."""
#         indices_set = {i if i >= 0 else len(self._prompt) + i for i in indices}
#         indices_set = {i for i in indices_set if 0 <= i < len(self._prompt)}
#         # Remove from prompt
#         self._prompt = [msg for idx, msg in enumerate(
#             self._prompt) if idx not in indices_set]
#         # Adjust history
#         self._history = [msg for msg in self._history if msg in self._prompt]

#     def reset(self) -> None:
#         """Reset the prompt to the base prompt and clear the history."""
#         self._prompt = copy.deepcopy(self._base_prompt)
#         self._history.clear()

#     def preprocess(self) -> List[dict[str, str]]:
#         """
#         Preprocess the prompt messages for the model.
#         Returns a list of dictionaries with 'role' and 'content'.
#         """
#         return [{'role': msg.role, 'content': msg.content} for msg in self._prompt]

#     def __repr__(self) -> str:
#         return f"{self.__class__.__name__}(prompt={pp.pformat(self._prompt)})"

#     def __str__(self) -> str:
#         return pp.pformat(self._prompt)

#     def copy_history(self, prompt: 'GSMQuestionPromptTemplate') -> None:
#         """
#         Copy the history from another GSMLlamaPromptTemplate instance.
#         If the prompt types differ, swap 'user' and 'assistant' roles in the copied history.
#         """
#         if not isinstance(prompt, GSMQuestionPromptTemplate):
#             raise TypeError(
#                 "The provided prompt must be an instance of GSMStrategyPromptTemplate.")

#         # Determine if role swapping is needed
#         if self._fsl_prompt_type != prompt._fsl_prompt_type:
#             # swapped_history = [
#             #     PromptMessage(
#             #         role='assistant' if msg.role == 'user' else 'user' if msg.role == 'assistant' else msg.role,
#             #         content=msg.content
#             #     )
#             #     for msg in prompt.history
#             # ]
#             pass
#         else:
#             swapped_history = copy.deepcopy(prompt.history)

#         # Update the current prompt and history
#         self._history.extend(swapped_history)
#         self._prompt.extend(swapped_history)

#     def add_eval(self,
#                  gold_trajectory: GSMLlamaPromptTemplate,
#                  mcts_prompt: GSMLlamaPromptTemplate,
#                  correct: bool
#                  ) -> None:
#         """ Adds the prompt from the leaf node of the optimal path """
#         ...
        
# if __name__=="__main__": 
#     path = '/lus/eagle/projects/FoundEpidem/bhsu/2024_research/agents/agents/data/gsm.jsonl'
#     dataset = read_jsonl_dataset(path)
    
#     prompt = GSMQuestionPromptTemplate()
#     breakpoint()
