"""Module for the vllm backend LLMGenerator."""

from __future__ import annotations

from typing import Literal
from enum import Enum
from vllm.sequence import Logprob
import torch
import numpy as np

from generators.utils import BaseConfig
from generators.generators.base_generator import BaseLLMGenerator, GeneratorOutput
from generators.prompts.chat_prompt import ChatMessageSequence

class ModelType(Enum):
    '''Suppored Models With VLLM'''
    FALCON7B = 'tiiuae/falcon-7b'
    FALCON40B = 'tiiuae/falcon-40b'
    GEMMATWO9B = 'google/gemma-2-9b'
    GEMMATWO27B = 'google/gemma-2-27b'
    LLAMA3INSTRUCT70B = 'meta-llama/Meta-Llama-3-70B-Instruct'
    LLAMA3170B = 'meta-llama/Meta-Llama-3.1-70B'
    LLAMA38B = 'meta-llama/Meta-Llama-3-8B-Instruct'
    MISTRAL7B = 'mistralai/Mistral-7B-Instruct-v0.1'
    MIXTRAL7X8B = 'mistralai/Mixtral-8x7B-Instruct-v0.1'
    PHI3MEDIUMINSTRUCT = 'microsoft/Phi-3-medium-128k-instruct'

class VLLMConfig(BaseConfig):
    """Configuration for the VLLMGenerator."""
    _name: Literal['vllm'] = 'vllm'  # type: ignore[assignment]
    # The name of the vllm LLM model, see
    # https://docs.vllm.ai/en/latest/models/supported_models.html
    llm_name: str = 'meta-llama/Meta-Llama-3-8B-Instruct'
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


class VLLMGenerator(BaseLLMGenerator, name='vllm', config=VLLMConfig):
    """Language model generator using vllm backend."""

    def __init__(self, 
                 llm_name: str, 
                 trust_remote_code: bool, 
                 temperature: float, 
                 min_p: float, 
                 top_p: float, 
                 max_tokens: int, 
                 use_beam_search: False, 
                 tensor_parallel_size: int, 
                 logprobs: int, 
                 use_tqdm: bool, 
                 dtype: str
                 ) -> None:
        """Initialize the VLLMGenerator.

        Parameters
        ----------
        config : vLLMGeneratorConfig
            The configuration for the VLLMGenerator.
        """
        from vllm import LLM
        from vllm import SamplingParams
        from transformers import AutoTokenizer
        
        assert llm_name in list(name.value for name in ModelType), f''' Model {llm_name} is not in supported models: {list(name.value for name in ModelType)}'''
        # Create the sampling params to use
        sampling_kwargs = {}
        if top_p:
            sampling_kwargs['top_p'] = top_p
        else:
            sampling_kwargs['min_p'] = min_p

        # Create the sampling params to use
        self.sampling_params = SamplingParams(
            temperature=temperature,
            max_tokens=max_tokens,
            logprobs=logprobs,
            # use_beam_search=config.use_beam_search,
            **sampling_kwargs,
        )
        # Create an LLM instance
        self.llm = LLM(
            model=llm_name,
            trust_remote_code=trust_remote_code, # NOTE: Fix to True 
            dtype=dtype,
            tensor_parallel_size=tensor_parallel_size,
        )

        # inference  attr
        self.use_tqdm = use_tqdm
        self.tokenizer = AutoTokenizer.from_pretrained(llm_name, 
                                                       trust_remote_code=trust_remote_code)
        self.max_tokens = self.tokenizer.model_max_length
        
    def generate(self, messages: dict | list[dict]) -> GeneratorOutput:
        """ 
        Generate response text from one prompt
        """
        try: 
            if isinstance(messages, dict): 
                messages = [messages]
            # validation 
            messages = ChatMessageSequence(messages).as_list() # validation, then convert to list
            output = self.llm.chat(messages=messages, 
                                    sampling_params=self.sampling_params,
                                    use_tqdm=self.use_tqdm)
            return GeneratorOutput(success=True, 
                                   output=output[0].outputs[0].text, 
                                   messages=messages)
        except Exception as e: 
            return GeneratorOutput(success=False, 
                                   output=f'Error: {e}', 
                                   messages=messages)
            
    def batch_generate(self, batch_messages: list[dict] | list[list[dict]]) -> list[GeneratorOutput]:
        """ 
        Generate response text from one prompt
        """
        if isinstance(batch_messages[0], dict): 
                batch_messages = [batch_messages]

        try: 
            batch_messages = [ChatMessageSequence(message).as_list() 
                              for message in batch_messages] # validation, then convert to list
            outputs = self.llm.chat(messages=batch_messages, 
                                    sampling_params=self.sampling_params,
                                    use_tqdm=self.use_tqdm)
            return [GeneratorOutput(success=True, 
                                    output=output.outputs[0].text, 
                                    messages=message)
                    for output, message in zip(outputs, batch_messages)]
            
        except Exception as e: 
            return [GeneratorOutput(success=False, 
                                   output=f'Error: {e}', 
                                   messages=messages) for messages in batch_messages]
        
        
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

        embeddings: list[float | torch.Tensor, np.ndarray] = [output.outputs[0].embedding
                                                              for output in outputs]

        return embeddings
    
    
if __name__=="__main__": 
    
    registry = BaseLLMGenerator.get_registery()
    generator_cls, config_cls = registry['vllm']
    generator: VLLMGenerator = generator_cls(**config_cls().model_dump())

    message = [
        {'role': 'system', 'content': 'You are an AI that will yell at me no matter what I do'},
        {'role': 'user', 'content': 'hello how are you doing today?'}, 
        {'role': 'assistant', 'content': 'I am doing well!'}, 
        {'role': 'assistant', 'content': 'what is the capital of France?'}
    ]
    message_2 = {'role': 'assistant', 'content': 'hello'}
    out = generator.generate(message)
    out_2 = generator.generate(message_2)
    
    batch_messages = [
    [
        {'role': 'system', 'content': 'You are an AI that will yell at me no matter what I do'},
        {'role': 'user', 'content': 'hello how are you doing today?'},
        {'role': 'assistant', 'content': 'I am doing well!'},
        {'role': 'assistant', 'content': 'What is the capital of France?'}
    ],
    [
        {'role': 'system', 'content': 'You are an AI that will yell at me no matter what I do'},
        {'role': 'user', 'content': 'Can you help me with a math problem?'},
        {'role': 'assistant', 'content': 'I CAN HELP YOU, BUT WHY CAN’T YOU SOLVE IT YOURSELF?!'},
        {'role': 'assistant', 'content': 'What is 2 + 2?'}
    ],
    [
        {'role': 'system', 'content': 'You are an AI that will yell at me no matter what I do'},
        {'role': 'user', 'content': 'Tell me a joke.'},
        {'role': 'assistant', 'content': 'WHY DO CHICKENS CROSS THE ROAD? TO GET AWAY FROM YOU!'},
        {'role': 'assistant', 'content': 'HAHAHA!'}
    ],
    [
        {'role': 'system', 'content': 'You are an AI that will yell at me no matter what I do'},
        {'role': 'user', 'content': 'What is the weather like today?'},
        {'role': 'assistant', 'content': 'IT’S SUNNY, BUT WHY DO YOU EVEN CARE?!'},
        {'role': 'assistant', 'content': 'GO OUTSIDE AND SEE FOR YOURSELF!'}
    ]
]
    batch_outputs = generator.batch_generate(batch_messages)