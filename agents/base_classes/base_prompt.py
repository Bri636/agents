"""Interface for all prompts to follow."""

from __future__ import annotations

from typing import Protocol
from abc import ABC, abstractmethod

from agents.configs import BaseConfig

class BasePromptTemplate(Protocol):
    """PromptTemplate protocol for all prompts to follow."""
    
    BASE_TYPE: str = "Prompts"

    def __init__(self, config: BaseConfig) -> None:
        """Initialize the prompt with the configuration."""
        ...

    def preprocess(
        self,
        text: str | list[str],
        contexts: list[list[str]] | None = None,
        scores: list[list[float]] | None = None,
    ) -> list[str]:
        """Preprocess the text into prompts.

        Parameters
        ----------
        text : str
            The text to preprocess.
        contexts : list[list[str]], optional
            The contexts to include for each text, by default None.
        scores : list[list[float]], optional
            The scores for each context, by default None.

        Returns
        -------
        list[str]
            The preprocessed prompts.
        """
        ...

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
        ...