"""Interface for all language model generators to follow."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Self, Tuple, Union, Protocol
from pydantic import BaseModel
from abc import ABC, abstractmethod

from agents.utils import BaseConfig
from agents.utils import ConfigLike
from agents.generators.chat_prompt import ChatMessageSequence

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
            