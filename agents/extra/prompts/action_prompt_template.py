from __future__ import annotations

'''Class for Implementing Strategy Prompts'''

from dataclasses import dataclass, field
from textwrap import dedent
from enum import Enum
from typing import Literal, Union, Optional, Any
from pydantic import BaseModel, Field

from agents.utils import BaseConfig
from agents.prompts.base_prompt_template import BasePromptTemplate
from agents import prompt_registry

PROMPT_NAME = 'Actor'

class Instructions(Enum): 
    ''' Instructions for parsing outputs'''
    COT = dedent('''
    Given a question, please decompose it into sub-questions. 
    For each sub-question, please answer it in a complete sentence, ending with \"The answer is\". 
    When the original question is answerable, please start the subquestion with \"Now we can answer the question: \".''')

class ActorInputPayload(BaseOutputPayload): 
    """ Outputs to present to the strategy
    :goal - str: overall goal that you are doing. for gsm8k, this is 'You are solving math problems'
    :instruction - str: any instructions that the LLM should do for parsing. for gsm8k, this it the COT instruction
    :task - str: description of task to do. for gsm8k, this is the question.
    :strategy - str: a high-level strategy produced by strategy chain. this is the mutated prompt
    """
    goal: str = "None"
    task: str = "None"
    strategy: str = "None"
    instruction: str = "None"
    
class ActorOutputPayload(BaseOutputPayload): 
    """ Output format """
    reasoning: str = "None"
    action: str = "None"
    
@prompt_registry.register(name=PROMPT_NAME, payloads={
    'input': ActorInputPayload
})
class ActorPromptTemplate(BasePromptTemplate):
    """Question answer prompt template."""

    template: str = dedent('''
Goal: 
=====
{goal}

Instructions: 
===========
{instruction}

Task:
=====
{task}

Strategies:
=========
{strategy}

''')

    def __init__(self, config: Optional[Any]=None) -> None:
        """Initialize the QuestionAnswerPromptTemplate."""
        self.config = config

    def preprocess(
        self,
        **kwargs
        ) -> str:
        """Preprocess the text into prompts.

        Parameters
        ----------
        text : str
            The text to format.

        Returns
        -------
        list[str]
            The formatted prompts.
        """
        # Build the prompts using the template
        return self.template.format(**kwargs)

    def postprocess(self, responses: list[str]) -> list[str]:
        """Postprocess the responses.

        Parameters
        ----------
        responses : list[str]
            The responses to postprocess.

        Returns
        -------
        list[str]
            The postprocessed responses.
        """
        # If present, remove the option number from the response
        responses = [
            r[3:] if r[:2] in ['1.', '2.', '3.', '4.'] else r
            for r in responses
        ]
        # If present, remove the period from the end of the response
        responses = [r if r and r[-1] != '.' else r[:-1] for r in responses]

        # Cast responses to lower caps in case model capitalized answers.
        responses = [r.lower() for r in responses]

        return responses