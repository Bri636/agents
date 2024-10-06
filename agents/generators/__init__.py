from __future__ import annotations

'''Import registrated generator classes'''

from typing import Union, Optional

from agents import generator_registry
from agents.registry.registry import import_submodules

from agents.generators.argo_generator import *

import_submodules(__name__) # import all files from generators

from agents.generators.argo_generator import ArgoGenerator, ArgoGeneratorConfig
from agents.generators.langchain_generator import LangChainGenerator, LangchainGeneratorConfig
from agents.generators.vllm_generator import VLLMGenerator, VLLMGeneratorConfig

Generators = Union[
    ArgoGenerator, 
    LangChainGenerator, 
    VLLMGenerator
]

GeneratorConfigs = Union[
    ArgoGeneratorConfig, 
    LangchainGeneratorConfig, 
    VLLMGeneratorConfig
]

STRATEGIES = {
    'argo': (ArgoGenerator, ArgoGeneratorConfig), 
    'langchain': (LangChainGenerator, LangchainGeneratorConfig), 
    'vllm': (VLLMGenerator, VLLMGeneratorConfig)
}

def get_generator(name: str, **kwargs: dict[str, Any]) -> BaseLLMGenerator:
    ''' Function for initializing generator '''
    
    strategy = STRATEGIES.get(name)  # type: ignore[arg-type]
    if not strategy:
        raise ValueError(
            f'Unknown generator name: {name}.'
            f' Available: {set(STRATEGIES.keys())}',
        )

    # Get the config and classes
    config_cls, cls = strategy
    return cls(config_cls(**kwargs))





