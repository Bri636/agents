"""Interface for all language model generators to follow."""

from __future__ import annotations

from typing import Protocol

from abc import ABC, abstractmethod

from agents.utils import BaseConfig

class BaseLLMGenerator(Protocol):
    """Generator protocol for all generators to follow."""

    def __init__(self, config: BaseConfig) -> None:
        """Initialize the generator with the configuration."""
        
    def generate(self, prompts: str | list[str]) -> list[str]:
        """Generate response text from prompts.

        list[str]
            A list of responses generated from the prompts
            (one response per prompt).
        """
        ...
    