from __future__ import annotations

'''Class for Implementing Strategy Prompts'''

from dataclasses import dataclass, field
from textwrap import dedent
from enum import Enum
from typing import Literal, Union, Optional
from pydantic import BaseModel

from agents.configs import BaseConfig
from agents.base_classes import BasePromptTemplate, BasePayload

class StrategyPromptTemplateConfig(BaseConfig):
    """Configuration for the StrategyPromptTemplate"""
    _name: Literal['strategy'] = 'strategy'  # type: ignore[assignment]

# class StrategyPayload(BasePayload): 
    
#     task: str = Fiel

class StrategyPromptTemplate(BasePromptTemplate):
    """Question answer prompt template."""

    template_with_context: str = dedent('''
                <s>[INST]
                You are an intelligent strategist agent. You will be given a overall task / environment / world that you have to solve.
                Come up with a plausable strategy for how you might want to navigate or solve your environment and
                help you reach the goal. Your response must be some kind of strategy or thinking style, even if you have to guess
                
                Environment or Task: 
                {environment_or_task}
                [/INST]
                ''')

    template_no_context: str = dedent('''
                <s>[INST]
                You are an intelligent strategist agent. You will be given a overall task / environment / world that you have to solve.
                Come up with a plausable strategy for how you might want to navigate or solve your environment and
                help you reach the goal. Your response must be some kind of strategy, even if you have to guess
                [/INST]
                ''')

    def __init__(self, config: StrategyPromptTemplateConfig) -> None:
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