""" Contains useful utils """

from __future__ import annotations

import json
from pathlib import Path
from typing import Literal
from typing import TypeVar
from typing import Union
from typing import TypeVar 

import yaml  # type: ignore[import-untyped]
from pydantic import BaseModel
from torch import Tensor
import torch

import logging
import sys
from typing import Optional

PathLike = Union[str, Path]

T = TypeVar('T')

class BaseConfig(BaseModel):
    """An interface to add JSON/YAML serialization to Pydantic models."""

    # A name literal to correctly identify and construct nested models
    # which have many possible options.
    _name: Literal[''] = ''

    def write_json(self, path: PathLike) -> None:
        """Write the model to a JSON file.

        Parameters
        ----------
        path : str
            The path to the JSON file.
        """
        with open(path, 'w') as fp:
            json.dump(self.model_dump(), fp, indent=2)

    @classmethod
    def from_json(cls: type[T], path: PathLike) -> T:
        """Load the model from a JSON file.

        Parameters
        ----------
        path : str
            The path to the JSON file.

        Returns
        -------
        T
            A specific BaseConfig instance.
        """
        with open(path) as fp:
            data = json.load(fp)
        return cls(**data)

    def write_yaml(self, path: PathLike) -> None:
        """Write the model to a YAML file.

        Parameters
        ----------
        path : str
            The path to the YAML file.
        """
        with open(path, 'w') as fp:
            yaml.dump(
                json.loads(self.model_dump_json()),
                fp,
                indent=4,
                sort_keys=False,
            )

    @classmethod
    def from_yaml(cls: type[T], path: PathLike) -> T:
        """Load the model from a YAML file.

        Parameters
        ----------
        path : PathLike
            The path to the YAML file.

        Returns
        -------
        T
            A specific BaseConfig instance.
        """
        with open(path) as fp:
            raw_data = yaml.safe_load(fp)
        return cls(**raw_data)

def batch_data(data: list[T], chunk_size: int) -> list[list[T]]:
    """Batch data into chunks of size chunk_size.

    Parameters
    ----------
    data : list[T]
        The data to batch.
    chunk_size : int
        The size of each batch.

    Returns
    -------
    list[list[T]]
        The batched data.
    """
    batches = [
        data[i * chunk_size : (i + 1) * chunk_size]
        for i in range(0, len(data) // chunk_size)
    ]
    if len(data) > chunk_size * len(batches):
        batches.append(data[len(batches) * chunk_size :])
    return batches


def calculate_returns(rewards: Tensor, gamma: float) -> Tensor:
    '''
    Inputs: 
    ======
    rewards: Tensor
        Shape (B, T = sequence length) of raw rewards

    Outputs: 
    =======
    rewards: Tensor 
        Shape (B, ) of raw returns from discounted sum 
    '''
    B, T = rewards.shape  # Get batch size (B) and trajectory length (T)
    # Initialize returns with the same shape as rewards
    returns = torch.zeros_like(rewards)

    for b in range(B):
        R = torch.tensor(0.0, device=rewards.device)
        for t in reversed(range(T)):
            R = rewards[b, t] + gamma * R
            returns[b, t] = R

    return returns[:, 0]  # Return the first return G_0 for each episode


# def configure_logger(level: str = 'info',
#                      logging_save_path: Optional[str] = None,
#                      rank: Optional[int] = None
#                      ) -> logging.Logger:
#     """Configure a logger to write logs and print statements to a file."""

#     LEVELS = {
#         'debug': logging.DEBUG,
#         'info': logging.INFO,
#         'warning': logging.WARNING,
#         'error': logging.ERROR,
#         'critical': logging.CRITICAL
#     }

#     # Determine the logger name
#     logger_name = f'Logger: {rank}' if rank is not None else 'Logger'

#     # Create a logger instance
#     logger = logging.getLogger(logger_name)
#     logger.setLevel(LEVELS.get(level.lower(), logging.DEBUG))  # Set the logging level

#     # Clear existing handlers
#     logger.handlers.clear()

#     # Create a logging formatter
#     formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')

#     # Add file handler if path is provided
#     if logging_save_path:
#         file_handler = logging.FileHandler(logging_save_path)
#         file_handler.setFormatter(formatter)
#         logger.addHandler(file_handler)

#     # Redirect stdout and stderr to the logger
#     class StreamToLogger:
#         """
#         Fake file-like stream object that redirects writes to a logger instance.
#         """
#         def __init__(self, logger, log_level):
#             self.logger = logger
#             self.log_level = log_level
#             self.buffer = ''

#         def write(self, message):
#             if message != '\n':
#                 self.buffer += message
#             if '\n' in message:
#                 self.flush()

#         def flush(self):
#             if self.buffer:
#                 self.logger.log(self.log_level, self.buffer.strip())
#                 self.buffer = ''

#     # Replace sys.stdout and sys.stderr with our stream to logger
#     sys.stdout = StreamToLogger(logger, logging.INFO)
#     sys.stderr = StreamToLogger(logger, logging.ERROR)

#     return logger
