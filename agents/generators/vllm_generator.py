"""Module for the vllm backend LLMGenerator."""

from __future__ import annotations

from typing import Literal
from enum import Enum
from vllm.sequence import Logprob
from agents.utils import BaseConfig
from agents.generators.base_generator import BaseLLMGenerator
import torch
import numpy as np


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
            model=config.llm_name,
            trust_remote_code=config.trust_remote_code, # NOTE: Fix to True 
            dtype=config.dtype,
            tensor_parallel_size=config.tensor_parallel_size,
        )

        # inference  attr
        self.use_tqdm = config.use_tqdm
        self.tokenizer = AutoTokenizer.from_pretrained(config.llm_name, 
                                                       trust_remote_code=config.trust_remote_code)
        self.max_tokens = self.tokenizer.model_max_length

    def prompt_exceeds_limit(self, prompts: dict[str, str] | list[dict[str, str]]) -> bool:
        """Counts the number of tokens in a prompt. If exceeds return True, else False.
        Note that the prompt is a list[dict[str, str]] or dict[str, str] that corresponds to the 
        openai chat format ie
        [
            {'role': ..., 
            'content': ...}, 
            ...  
        ]
        """
        # Ensure that the prompts are in a list
        if isinstance(prompts, dict):
            prompts = [prompts]
        # Concatenate messages into a text string
        text = ''
        for message in prompts:
            role = message.get('role', '')
            content = message.get('content', '')
            text += f"{role}:\n{content}\n"
        input_ids = self.tokenizer.encode(text)
        num_tokens = len(input_ids)
        max_context_length = self.tokenizer.model_max_length

        return bool(num_tokens > max_context_length)

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
                                                                                              list[list[str]],
                                                                                              list[list[float]]]:
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
        log_probs: list[dict[int, Logprob]] = [
            output.outputs[0].logprobs for output in outputs]
        
        log_prob_seqs: list[list[float]] = [self.extract_log_probs(log_prob)['log_probs'] 
                                           for log_prob in log_probs]
        token_seqs: list[list[str]] = [self.extract_log_probs(log_prob)['tokens'] 
                                           for log_prob in log_probs]
        # token_seq, log_prob_seq = self.extract_log_probs(log_probs).values()
        return {'text': responses,
                'token_seq': token_seqs,
                'log_probs': log_prob_seqs,
                }

    def extract_log_probs(self, log_probs: list[dict[str, Logprob]]) -> dict[list[str], list[float]]:
        """ processes through the log_probs objects to return a sequence of the log probs and the sequence of text """

        token_seq = []
        log_prob_seq = []
        for log_prob_dict in log_probs:
            log_prob_obj: Logprob = next(
                iter(log_prob_dict.values()))  # extract logprobs object
            log_prob, token = log_prob_obj.logprob, log_prob_obj.decoded_token
            token_seq.append(token)
            log_prob_seq.append(log_prob)

        return {
            'tokens': token_seq,
            'log_probs': log_prob_seq
        }

    def embed(self, prompts:  dict[str, str] | list[dict[str, str]]) -> list[str]:

        if isinstance(prompts, dict):
            prompts = [prompts]

        outputs = self.llm.encode(prompts=prompts,
                                #   sampling_params=self.sampling_params,
                                  use_tqdm=self.use_tqdm)
        breakpoint()
        embeddings: list[float | torch.Tensor, np.ndarray] = [output.outputs[0].embedding
                                                              for output in outputs]
        breakpoint()
        return embeddings


if __name__ == "__main__":

    from agents.gsm8k.utils import read_jsonl_dataset, batch_sample_gsm
    # from agents.prompts.gsm_llama_prompts ...

    data_path = '/lus/eagle/projects/FoundEpidem/bhsu/2024_research/agents/agents/data/gsm.jsonl'
    batch_size = 16

    dataset = read_jsonl_dataset(data_path)
    samples = batch_sample_gsm(dataset, batch_size)

    breakpoint()
