from __future__ import annotations

'''Class for Implementing Strategy Prompts'''

from dataclasses import dataclass, field
from textwrap import dedent
from enum import Enum
from typing import Literal, Union, Optional, Any
from pydantic import BaseModel, Field

from agents.utils import BaseConfig
from agents.prompts.base_prompt import BasePromptTemplate, BaseInputPayload, BaseOutputPayload
from agents import prompt_registry

PROMPT_NAME = 'Strategist'

class StrategyInputPayload(BaseOutputPayload): 
    """ Outputs to present to the strategy """
    task: str
    context: str
    
class StrategyOutputPayload(BaseOutputPayload): 
    """ Output format """
    strategy: str 
    
@prompt_registry.register(name=PROMPT_NAME, payloads={
    'input': StrategyInputPayload, 
    'output': StrategyOutputPayload
})
class StrategyPromptTemplate(BasePromptTemplate):
    """Question answer prompt template."""

    template_with_context: str = dedent('''
You are an intelligent strategist agent. You will be given an overall task or environment that you have to solve.
Come up with a plausable strategy for how you might want to navigate or solve your environment and
help you reach the goal. Your response must be some kind of strategy or thinking style, even if you have to guess. 
Keep the strategy 

Environment or Task:
=================== 
{task}

Context: 
========
{context}
''')

    def __init__(self, config: Optional[Any]=None) -> None:
        """Initialize the QuestionAnswerPromptTemplate."""
        self.config = config

    def _format_prompt(
        self,
        environment_or_task:Optional[str | list[str]]
        ) -> str:
        """Format the prompt with the question and context."""
        
        return self.template_with_context.format(
            environment_or_task=environment_or_task
        )

    def preprocess(
        self,
        environment_or_task:Optional[str | list[str]]
        ) -> list[str]:
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
        # Ensure text is a list
        if isinstance(environment_or_task, str):
            environment_or_task = [environment_or_task]

        # If no contexts are provided, use the no-context template
        if environment_or_task is None:
            return [self.template_no_context] * len(environment_or_task)

        # Build the prompts using the template
        return list(map(self._format_prompt, environment_or_task))

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