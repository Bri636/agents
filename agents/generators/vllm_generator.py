"""Module for the vllm backend LLMGenerator."""

from __future__ import annotations

from typing import Literal, Union
from enum import Enum
from vllm.sequence import Logprob
import torch
import numpy as np

from agents.utils import BaseConfig
from agents.generators.base_generator import BaseLLMGenerator

ChatMessage = Union[dict[str, str], list[dict[str, str]]]
""" 
A single OpenAI-like chat message {'role': ..., 'content': ...}. 
If it is a list of {'role': ..., 'content': ...}, then it contains the chat histories.
"""
LogProbs = Union[np.ndarray, list[float]]
""" N x 1 array of lob probs corresponding to ONE output sequence """

class ModelType(Enum):
    '''Suppored Models With VLLM'''
    LLAMA3INSTRUCT70B = 'meta-llama/Meta-Llama-3-70B-Instruct'
    LLAMA3170B = 'meta-llama/Meta-Llama-3.1-70B'
    LLAMA38B = 'meta-llama/Meta-Llama-3-8B-Instruct'

class VLLMGeneratorConfig(BaseConfig):
    """Configuration for the VLLMGenerator."""
    _name: Literal['vllm'] = 'vllm'  # type: ignore[assignment]
    trust_remote_code: bool = True
    """Whether to trust remote code."""
    temperature: float = 0.5
    """Temperature for sampling."""
    min_p: float = 0.1
    """Min p for sampling."""
    top_p: float = 0.0
    """Top p for sampling (off by default)."""
    max_tokens: int = 2000
    """Max tokens to generate."""
    use_beam_search: bool = False
    """Whether to use beam search."""
    tensor_parallel_size: int = 1
    """The number of GPUs to use."""
    logprobs: int = 1
    """Number of log probabilities to return per output token."""
    use_tqdm: bool = False
    """Whether to use tqdm during inference."""
    dtype: str = 'float16'
    """Data type for computations (e.g., 'float16')."""

class VLLMGenerator(BaseLLMGenerator):
    """Language model generator using vllm backend."""

    def __init__(self, model_name_or_path: str, config: VLLMGeneratorConfig) -> None:
        """Initialize the VLLMGenerator.

        Parameters
        ----------
        config : vLLMGeneratorConfig
            The configuration for the VLLMGenerator.
        """
        from vllm import LLM
        from vllm import SamplingParams
        from transformers import AutoTokenizer

        # Create the sampling params to use
        sampling_kwargs = {}
        if config.top_p:
            sampling_kwargs['top_p'] = config.top_p
        else:
            sampling_kwargs['min_p'] = config.min_p

        # Create the sampling params to use
        self.sampling_params = SamplingParams(
            temperature=config.temperature,
            max_tokens=config.max_tokens,
            logprobs=config.logprobs,
            # use_beam_search=config.use_beam_search,
            **sampling_kwargs,
        )
        # Create an LLM instance
        self.llm = LLM(
            model=model_name_or_path,
            trust_remote_code=config.trust_remote_code, # NOTE: Fix to True 
            dtype=config.dtype,
            tensor_parallel_size=config.tensor_parallel_size,
        )

        # inference  attr
        self.use_tqdm = config.use_tqdm
        self.tokenizer = AutoTokenizer.from_pretrained(model_name_or_path, trust_remote_code=config.trust_remote_code)
        self.max_tokens = self.tokenizer.model_max_length

    def prompt_exceeds_limit(self, prompts: ChatMessage) -> bool:
        """
        Counts the number of tokens in a prompt. If exceeds return True, else False.
        """
        # Ensure that the prompts are in a list
        if isinstance(prompts, dict):
            prompts = [prompts]
        text = ''
        for message in prompts:
            role = message.get('role', '')
            content = message.get('content', '')
            text += f"{role}:\n{content}\n"
        input_ids = self.tokenizer.encode(text)
        num_tokens = len(input_ids)
        max_context_length = self.tokenizer.model_max_length
        return bool(num_tokens > max_context_length)

    def generate(self, prompts: ChatMessage) -> list[str]:
        """Generate response text from prompts.

        Parameters
        ----------
        prompts : dict[str, str] | list[dict[str, str]]
            The prompts to generate text from, of form: 
            [{'user': ..., 'content': ...}, ...]

        Returns
        -------
        list[str]
            A list of responses generated from the prompts
            (one response per prompt).
        """
        if isinstance(prompts, dict):
            prompts = [prompts]
        outputs = self.llm.chat(messages=prompts,
                                sampling_params=self.sampling_params,
                                use_tqdm=self.use_tqdm)
        responses: list[str] = [output.outputs[0].text
                                for output in outputs]
        return responses
    
    def _extract_log_probs(self, log_probs: list[dict[str, Logprob]]) -> LogProbs:
        """ processes through the log_probs objects to return a sequence of the log probs """
        log_prob_seq = []
        for log_prob_dict in log_probs:
            log_prob_obj: Logprob = next(iter(log_prob_dict.values()))  # extract logprobs object
            log_prob = log_prob_obj.logprob
            log_prob_seq.append(log_prob)
        return log_prob_seq

    def generate_with_logprobs(self, prompts: ChatMessage) -> dict[list[str], list[LogProbs]]:
        """Generate response text from prompts.

        Parameters
        ----------
        prompts: dict[str, str] | list[dict[str, str]]
            The prompts to generate text from, of form: 
            [{'user': ..., 'content': ...}, ...]

        Returns
        -------
        dict[str: list[str] | list[list[float]]]
            Dictionary that contains the batched texts, and the sequence of logprobs
        """
        # Ensure that the prompts are in a list
        if isinstance(prompts, dict):
            prompts = [prompts]
        outputs = self.llm.chat(messages=prompts, 
                                sampling_params=self.sampling_params, 
                                use_tqdm=self.use_tqdm)
        responses: list[str] = [output.outputs[0].text 
                                for output in outputs]
        log_probs: list[dict[int, Logprob]] = [output.outputs[0].logprobs 
                                               for output in outputs]
        log_prob_seqs: list[np.ndarray] = [np.array(self._extract_log_probs(log_prob))
                                           for log_prob in log_probs]
        return {'text': responses, 'log_probs': log_prob_seqs}