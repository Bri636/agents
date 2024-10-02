"""Interface for all payloads for follow."""

from __future__ import annotations

from typing import Protocol
from abc import ABC, abstractmethod
from pydantic import BaseModel, field_validator

from agents.configs import BaseConfig

class BasePayload(BaseModel, Protocol): 
    '''Container for inputs into a model'''
    ...
    

