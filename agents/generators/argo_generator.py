import os
import requests
import json
from typing import Any, List, Literal, Mapping, Optional, Tuple, Dict
from langchain_core.callbacks.manager import CallbackManagerForLLMRun
from langchain_core.language_models.llms import LLM
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import JsonOutputParser
import requests
import json
import os
from pydantic import Field, BaseModel
from collections import defaultdict
from enum import Enum
from dataclasses import asdict

from langchain_core.embeddings.embeddings import Embeddings
from sqlalchemy import desc
from langchain_core.prompts import PromptTemplate

class ModelType(Enum):
    GPT35 = 'gpt35'
    GPT4 = 'gpt4'
    GPT4O = 'gpt4o'
    GPT4TURBO = 'gpt4turbo'
    
class ArgoGeneratorConfig(BaseConfig): 
    '''Base Config for Argo Language Model'''
    model_type:Literal['gpt35', 'gpt4', 'gpt4o', 'gpt4turbo', 'o1preview']=Field(
        default='gpt4turbo', 
        description='What kind of language model to use from openai'
    )
    url:str=Field(
        default="https://apps-dev.inside.anl.gov/argoapi/api/v1/resource/chat/", 
        description='URL for pointing model at'
    )
    temperature:float=Field(
        default=0.00001, 
        description='What temperature of the model to be at for sampling'
    )
    system:Optional[str]=Field(
        default=None, 
        description='What system to use'
    )
    top_p:float=Field(
        default=0.0000001, 
        description='What top_p to use for sampling'
    )
    user:str=Field(
        default='bhsu'
    )
    
class ArgoLLM(LLM):

    model_type: ModelType = ModelType.GPT4
    url: str = "https://apps-dev.inside.anl.gov/argoapi/api/v1/resource/chat/"
    temperature: Optional[float] = 0.0
    system: Optional[str]
    top_p: Optional[float]= 0.0000001
    user: str = 'bhsu'
    
    @property
    def _llm_type(self) -> str:
        return "ArgoLLM"

    def _call(
        self,
        prompt: str,
        stop: Optional[List[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> str:

        headers = {
            "Content-Type": "application/json"
        }
        params = {
            **self._get_model_default_parameters,
            **kwargs,
            "prompt": [prompt],
            "stop": []
        }

        params_json = json.dumps(params);
        response = requests.post(self.url, headers=headers, data=params_json)

        if response.status_code == 200:
            parsed = json.loads(response.text)
            return parsed['response']
        else:
            raise Exception(f"Request failed with status code: {response.status_code} {response.text}")

    @property
    def _get_model_default_parameters(self):
        return {
            "user": self.user,
            "model": self.model,
            "system": "" if self.system is None else self.system,
            "temperature": self.temperature,
            "top_p":  self.top_p
        }

    @property
    def model(self):
        return self.model_type
    
    @property
    def _identifying_params(self) -> Mapping[str, Any]:
        """Get the identifying parameters."""
        return {}

    @property
    def _generations(self):
        return
    
    
class ArgoGenerator(LLMGenerator): 
    '''Argo generator for generating outputs'''
    
    def __init__(self, config:ArgoGeneratorConfig): 
        from langchain.chains.llm import LLMChain
        from langchain_core.prompts import ChatPromptTemplate
        
        llm = ArgoLLM()
        
        # set attributes of model 
        setattr(llm, 'model_type', config.model_type)
        setattr(llm, 'temperature', config.temperature)
        setattr(llm, 'system', config.system)
        setattr(llm, 'top_p', config.top_p)
        setattr(llm, 'user', config.user)
        
        prompt = ChatPromptTemplate.from_template('{input}')
        
        chain = LLMChain(llm=llm, 
                         prompt=prompt)
        
        self.llm = llm
        self.chain = chain
        
    def generate(self, prompts: str | list[str]) -> str | list[str]: 
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
        