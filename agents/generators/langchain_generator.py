from __future__ import annotations

import numpy as np
from dotenv import load_dotenv
from typing import Any, List, Literal, Union
from pydantic import Field
from pathlib import Path
from enum import Enum

from agents.configs import BaseConfig
from agents.generators import BaseLLMGenerator
from agents.generators import generator_registry

class ModelType(Enum): 
    GPT35 = 'gpt-3.5-turbo'
    GPT4 = 'gpt-4'
    GPT4O = 'gpt-4o'
    GPT4TURBO = 'gpt-4-turbo'
    CLAUDE3OPUS = 'claude-3-opus-20240229'
    GEMINIPRO = 'gemini-pro'

class LangchainGeneratorConfig(BaseConfig):
    '''Base Config for Langchain Model'''
    model_name: Literal['gpt-3.5-turbo',
                        'gpt-4',
                        'gpt-4-turbo',
                        'gpt-4-turbo-preview',
                        'gpt-4o',
                        'claude-3-opus-20240229',
                        'gemini-pro'] = Field(
        'gpt-4o',
        description='what kind of llm to use'
    )
    temperature: float = Field(
        default=0.0,
        description='What temperature to use for sampling'
    )
    verbose: bool = Field(
        default=True,
        description='whether or not the langchain llm should be verbose or not'
    )
    dotenv_path: Path = Field(
        default=Path.home() / '.env',
        description='Path to the .env file. Contains API keys: '
        'OPENAI_API_KEY, GOOGLE_API_KEY, ANTHROPIC_API_KEY',
    )

generator_registry.register(BaseLLMGenerator.CLASS_TYPE, LangchainGeneratorConfig)
class LangChainGenerator(BaseLLMGenerator):
    """Create simple language chains for inference."""

    def __init__(self, config: LangchainGeneratorConfig) -> None:
        """Initialize the LangChainGenerator."""
        from langchain.chains.llm import LLMChain
        from langchain_community.chat_models import ChatOpenAI
        from langchain_anthropic import ChatAnthropic
        from langchain_core.prompts import ChatPromptTemplate
        from langchain_google_genai import GoogleGenerativeAI

        # Load environment variables from .env file containing
        # API keys for the language models
        load_dotenv(config.dotenv_path)
        # Define the possible chat models
        chat_models = {
            'gpt-3.5-turbo': ChatOpenAI,
            'gpt-4': ChatOpenAI,
            'gpt-4-turbo': ChatOpenAI,
            'gpt-4-turbo-preview': ChatOpenAI,
            'gpt-4o': ChatOpenAI,
            'gemini-pro': GoogleGenerativeAI,
            'claude-3-opus-20240229': ChatAnthropic,
        }
        # Get the chat model based on the configuration
        chat_model = chat_models.get(config.model_name)
        if not chat_model:
            raise ValueError(f'Invalid chat model: {config.llmconfig.model}')

        llm = ChatOpenAI(model=config.model_name,
                         temperature=config.temperature,
                         verbose=config.verbose)
        
        # Create the prompt template (input only)
        prompt = ChatPromptTemplate.from_template('{input}')
        
        chain = LLMChain(
            llm=llm,
            prompt=prompt,
        )
        
        self.llm = llm
        self.chain = chain
    
    def generate(self, prompts: Union[str, list[str]]) -> list[str]: 
        '''Generated function for list of payload inferences'''
        
        if isinstance(prompts, str): 
            prompts = [prompts]
            
        inputs = [{'input': prompt} for prompt in prompts]
        raw_outputs = self.chain.batch(inputs)
        outputs = [output['text'] for output in raw_outputs]
        
        return outputs
        
    def __str__(self):
        return f"Langchain Generator with Chain: {self.llm}"

    def __repr__(self):
        return f"Langchain Generator with Chain: {self.llm}"

