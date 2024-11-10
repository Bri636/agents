"""Module for the vllm backend LLMGenerator."""

from __future__ import annotations

from typing import Literal
from enum import Enum
from vllm.sequence import Logprob
from agents.utils import BaseConfig
from agents.generators.base_generator import BaseLLMGenerator


class ModelType(Enum):
    '''Suppored Models With VLLM'''

    # FALCON7B = 'tiiuae/falcon-7b'
    # FALCON40B = 'tiiuae/falcon-40b'
    # GEMMATWO9B = 'google/gemma-2-9b'
    # GEMMATWO27B = 'google/gemma-2-27b'
    LLAMA3INSTRUCT70B = 'meta-llama/Meta-Llama-3-70B-Instruct'
    LLAMA3170B = 'meta-llama/Meta-Llama-3.1-70B'
    LLAMA38B = 'meta-llama/Meta-Llama-3-8B-Instruct'
    MISTRAL7B = 'mistralai/Mistral-7B-Instruct-v0.1'
    MIXTRAL7X8B = 'mistralai/Mixtral-8x7B-Instruct-v0.1'
    # PHI3MEDIUMINSTRUCT = 'microsoft/Phi-3-medium-128k-instruct'


class VLLMGeneratorConfig(BaseConfig):
    """Configuration for the VLLMGenerator."""
    _name: Literal['vllm'] = 'vllm'  # type: ignore[assignment]
    # The name of the vllm LLM model, see
    # https://docs.vllm.ai/en/latest/models/supported_models.html
    llm_name: str = ModelType.LLAMA38B.value
    # Whether to trust remote code
    trust_remote_code: bool = True
    # Temperature for sampling
    temperature: float = 0.5
    # Min p for sampling
    min_p: float = 0.1
    # Top p for sampling (off by default)
    top_p: float = 0.0
    # Max tokens to generate
    max_tokens: int = 2000
    # Whether to use beam search
    use_beam_search: bool = False
    # The number of GPUs to use
    tensor_parallel_size: int = 1
    # number of log probs to return per output token
    logprobs: int = 1
    # whether to use tqdm during inference
    use_tqdm: bool = False
    dtype: str = 'float16'


class VLLMGenerator(BaseLLMGenerator):
    """Language model generator using vllm backend."""

    def __init__(self, config: VLLMGeneratorConfig) -> None:
        """Initialize the VLLMGenerator.

        Parameters
        ----------
        config : vLLMGeneratorConfig
            The configuration for the VLLMGenerator.
        """
        from vllm import LLM
        from vllm import SamplingParams

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
            model=config.llm_name,
            trust_remote_code=config.trust_remote_code,
            dtype=config.dtype,
            tensor_parallel_size=config.tensor_parallel_size,
        )
        
        # inference  attr 
        self.use_tqdm = config.use_tqdm

    def generate(self, prompts:  dict[str, str] | list[dict[str, str]]) -> list[str]:
        """Generate response text from prompts.

        Parameters
        ----------
        prompts : dict[str, str] | list[dict[str, str]]
            The prompts to generate text from, of form: 
            [
                {'user': ..., 
                'content': ...}, 
                ...  
            ]

        Returns
        -------
        list[str]
            A list of responses generated from the prompts
            (one response per prompt).
        """
        # Ensure that the prompts are in a list
        if isinstance(prompts, dict):
            prompts = [prompts]

        outputs = self.llm.chat(messages=prompts,
                                sampling_params=self.sampling_params,
                                use_tqdm=self.use_tqdm)
        responses: list[str] = [output.outputs[0].text
                                for output in outputs]

        return responses
        
    def generate_with_logprobs(self, prompts:  dict[str, str] | list[dict[str, str]]) -> dict[list[str],
                                                                                list[str],
                                                                                list[float]]:
        """Generate response text from prompts.

        Parameters
        ----------
        prompts : dict[str, str] | list[dict[str, str]]
            The prompts to generate text from, of form: 
            [
                {'user': ..., 
                'content': ...}, 
                ...  
            ]

        Returns
        -------
        list[str]
            A list of responses generated from the prompts
            (one response per prompt).
        """
        # Ensure that the prompts are in a list
        if isinstance(prompts, dict):
            prompts = [prompts]

        outputs = self.llm.chat(messages=prompts,
                                sampling_params=self.sampling_params,
                                use_tqdm=True)
        responses: list[str] = [output.outputs[0].text
                                for output in outputs]
        log_probs: list[dict[int, Logprob]] = [output.outputs[0].logprobs
                                               for output in outputs]

        token_seq, log_prob_seq = self.extract_log_probs(log_probs).values()

        return {'text': responses,
                'token_seq': token_seq,
                'log_probs': log_prob_seq,
                }
    
    def extract_log_probs(self, log_probs: list[dict[str, Logprob]]) -> dict[list[str], list[float]]:
        """ processes through the log_probs objects to return a sequence of the log probs and the sequence of text """

        token_seq = []
        log_prob_seq = []
        for log_prob_dict in log_probs:
            log_prob_obj: Logprob = log_prob_dict.values()
            log_prob, token = log_prob_obj.logprob, log_prob_obj.decoded_token
            token_seq.append(token)
            log_prob_seq.append(log_prob)

        return {
            'tokens': token_seq,
            'log_probs': log_prob_seq
        }
