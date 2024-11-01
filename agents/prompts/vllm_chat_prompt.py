from __future__ import annotations

'''Class for Implementing Strategy Prompts'''

from dataclasses import dataclass, field
from textwrap import dedent
from enum import Enum
from typing import Literal, Union, Optional, Any
from pydantic import BaseModel, Field

from agents.utils import BaseConfig
from agents.prompts.base_prompt_template import BasePromptTemplate, BaseInputPayload, BaseOutputPayload
from agents import prompt_registry

PROMPT_NAME = 'vllm'
    
@prompt_registry.register(name=PROMPT_NAME)
class StrategyPromptTemplate(BasePromptTemplate):
    """ Question answer prompt template."""

    template: str = dedent('''
                           
                           
''')

    def __init__(self, config: Optional[Any]=None) -> None:
        """Initialize the QuestionAnswerPromptTemplate."""
        self.config = config

    def preprocess(
        self,
        **kwargs
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
    
    
if __name__ == "__main__": 
    
    breakpoint()