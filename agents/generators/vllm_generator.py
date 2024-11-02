"""Module for the vllm backend LLMGenerator."""

from __future__ import annotations

from typing import Literal
from enum import Enum

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
            # use_beam_search=config.use_beam_search,
            **sampling_kwargs,
        )

        # Create an LLM instance
        self.llm = LLM(
            model=config.llm_name,
            trust_remote_code=config.trust_remote_code,
            dtype='bfloat16',
            tensor_parallel_size=config.tensor_parallel_size,
        )

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
                                use_tqdm=True)
        responses = [output.outputs[0].text for output in outputs]

        return responses

        # # Generate responses from the prompts. The output is a list of
        # # RequestOutput objects that contain the prompt, generated text,
        # # and other information.
        # outputs = self.llm.generate(prompts, self.sampling_params)

        # # Extract the response from the outputs
        # responses = [output.outputs[0].text for output in outputs]

        # return responses