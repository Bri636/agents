from typing import Union
from agents.generators.base_generator import BaseLLMGenerator
from agents.generators.vllm_generator import VLLMGenerator, VLLMGeneratorConfig

Generator=Union[BaseLLMGenerator, VLLMGenerator]