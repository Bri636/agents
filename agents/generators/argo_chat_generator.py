import os
import requests
import json
from typing import Any, List, Literal, Mapping, Optional, Tuple, Dict, Union
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

from langchain.chat_models import ChatOpenAI
from langchain.schema import HumanMessage, SystemMessage, AIMessage
from typing import List, Dict

from agents.utils import BaseConfig
from agents.generators.base_generator import BaseLLMGenerator

class ModelType(Enum):
    '''Suppored Models With Argo'''
    GPT35 = 'gpt35'
    GPT4 = 'gpt4'
    GPT4O = 'gpt4o'
    GPT4TURBO = 'gpt4turbo'
    
class ArgoGeneratorConfig(BaseConfig): 
    '''Base Config for Argo Language Model'''
    
    _name: str = 'ArgoGenerator'
    
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
    logprobs: bool = Field(
        default=False, 
        description='Whether to return logprobs or not'
    )
    
class ArgoLLM(LLM):
    '''Overwritten langchain LLM model'''
    model_type: ModelType = ModelType.GPT4
    url: str = "https://apps-dev.inside.anl.gov/argoapi/api/v1/resource/chat/"
    temperature: Optional[float] = 0.0
    system: Optional[str] = 'None'
    top_p: Optional[float]= 0.0000001
    user: str = 'bhsu'
    logprobs: bool = False
    
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

    
class LangChainFSLGenerator(BaseLLMGenerator): 
    '''Argo generator for generating outputs'''
    
    def __init__(self, config: ArgoGeneratorConfig) -> None: 
        from langchain_openai import ChatOpenAI
        
        from langchain.chains.llm import LLMChain
        from langchain_core.prompts import ChatPromptTemplate
        
        llm = ArgoLLM()
        
        # set class attributes of model 
        setattr(llm, 'model_type', config.model_type)
        setattr(llm, 'temperature', config.temperature)
        setattr(llm, 'system', config.system)
        setattr(llm, 'top_p', config.top_p)
        setattr(llm, 'user', config.user)
        setattr(llm, 'logprobs', config.logprobs)
        
        prompt = ChatPromptTemplate.from_template('{input}')
        
        chain = LLMChain(llm=llm, prompt=prompt)
        
        self.llm = llm
        self.chain = chain
        
    def generate(self, prompts: Union[Dict[str, str], List[Dict[str, str]]]) -> List[str]:
        """Generate output from a single dict or list of dicts in OpenAI message format."""
        
        # Ensure prompts is a list
        if isinstance(prompts, dict):
            prompts = [prompts]
        
        # Concatenate messages with roles
        def concatenate_messages_with_roles(messages: List[Dict[str, str]]) -> str:
            return "\n".join(f"{msg['role']}:\n{msg['content']}" 
                             for msg in messages if 'content' in msg and 'role' in msg)

        # Convert prompts into a single formatted string with roles
        concatenated_prompt = concatenate_messages_with_roles(prompts)
        
        # Call LangChain's LLMChain with the concatenated prompt
        response = self.chain.run(input=concatenated_prompt)
        
        return [response]
    
    
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
    
    # def extract_log_probs(self, log_probs: list[dict[str, Logprob]]) -> dict[list[str], list[float]]:
    #     """ processes through the log_probs objects to return a sequence of the log probs and the sequence of text """

    #     token_seq = []
    #     log_prob_seq = []
    #     for log_prob_dict in log_probs:
    #         log_prob_obj: Logprob = log_prob_dict.values()
    #         log_prob, token = log_prob_obj.logprob, log_prob_obj.decoded_token
    #         token_seq.append(token)
    #         log_prob_seq.append(log_prob)

    #     return {
    #         'tokens': token_seq,
    #         'log_probs': log_prob_seq
    #     }
        
        
if __name__ == "__main__": 
    
    import requests
    import json

    # API endpoint to POST
    url = "https://apps-dev.inside.anl.gov/argoapi/api/v1/resource/chat/"

    # Data to be sent as a POST in JSON format
    data = {
        "model": ModelType.GPT4TURBO.value,
        "user": "bhsu",
        "prompt": ["What is your name", "What is your favorite color?"],
        "logprobs": True
    }
    
    #     model_type: ModelType = ModelType.GPT4
    # url: str = "https://apps-dev.inside.anl.gov/argoapi/api/v1/resource/chat/"
    # temperature: Optional[float] = 0.0
    # system: Optional[str] = 'None'
    # top_p: Optional[float]= 0.0000001
    # user: str = 'bhsu'
    # logprobs: bool = False
    
    data = {
            "user": "bhsu",
            "model": ModelType.GPT4TURBO.value,
            "system": "",
            "temperature": 0,
            "top_p":  0.0000001, 
            "logprobs": True,
            "prompt": ['hello how are you doing?'],
            "stop": []
        }

    # Convert the dict to JSON
    payload = json.dumps(data)

    # Adding a header stating that the content type is JSON
    headers = {"Content-Type": "application/json"}

    # Send POST request
    response = requests.post(url, data=payload, headers=headers)

    # Receive the response data
    print("Status Code:", response.status_code)
    print("JSON Response ", response.json())


    # Usage
    llm = ArgoLLM()
    llm.logprobs = True  # Enable logprobs if needed
    response, logprobs = llm._call(prompt="Hi how are you?")

    
    output = generator.generate([{'role': 'user', 'content': 'hi how are you doing?'}, 
                                 {'role': 'assistant', 'content': 'I am doing fine, how are you doing?'}])
    
    breakpoint()