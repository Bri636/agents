from __future__ import annotations

import numpy as np
from dotenv import load_dotenv
from typing import Any, List, Literal, Union
from pydantic import Field
from pathlib import Path
from enum import Enum

from agents.configs import BaseConfig

class LangchainEmbedderConfig(BaseConfig):
    '''Base Config for Langchain Embedding Models'''
    model_name: Literal['text-embedding-ada-002',
                        'text-embedding-3-small',
                        'text-embedding-3-large',
                        'voyage-2',
                        'voyage-large-2-instruct',
                        'voyage-code-2',
                        'voyage-large-2'
                        ] = Field(
        'text-embedding-ada-002',
        description='what kind of llm to use'
    )
    dimensions: int = Field(
        default=768,
        description='What dimension to use for embeddings'
    )
    chunk_size: int = Field(
        default=2048,
        description='Chunk Size used for embedding'
    )
    show_progress_bar: bool = Field(
        default=True,
        description='Whether to show progress bar for embeddings or not'
    )
    dotenv_path: Path = Field(
        default=Path.home() / '.env',
        description='Path to the .env file. Contains API keys: '
        'OPENAI_API_KEY, GOOGLE_API_KEY, ANTHROPIC_API_KEY',
    )


class LangChainEmbedder:
    '''Embedder class for langchain'''

    def __init__(self, config: LangchainEmbedderConfig) -> None:
        from sklearn.metrics.pairwise import cosine_similarity
        import pandas as pd
        import numpy as np
        from langchain_openai import OpenAIEmbeddings
        from langchain_voyageai import VoyageAIEmbeddings

        load_dotenv(config.dotenv_path)

        embedder_models = {
            'text-embedding-3-small': OpenAIEmbeddings,
            'text-embedding-3-large': OpenAIEmbeddings,
            'text-embedding-ada-002': OpenAIEmbeddings,
            'voyage-large-2': VoyageAIEmbeddings,
            'voyage-code-2': VoyageAIEmbeddings,
            'voyage-2': VoyageAIEmbeddings,
        }

        embedder_model = embedder_models.get(config.model_name)
        embedder_model(model=config.model_name, dimensions=config.dimensions,
                    chunk_size=config.chunk_size, show_progress_bar=config.show_progress_bar)


        rag_systems = {
            'qdrant': ''
        }
        # rag_system=rag_systems.get(config.ragconfig.model)
        
        self.embedder_model=embedder_model

    def create_memory(self):

        return None

    def embed_query(self, input: str, array: bool = False) -> List[float]:
        if array:
            return np.array(self.embed_model.embed_query(input))
        return self.embed_model.embed_query(input)

    def embed_documents(self, inputs: list[Any]):

        return None

    def store_document(self):
        return None

    def retrieve_documents(self):
        return None

    def __repr__(self):
        return f'Embedding Model is type: {self.embed_model}'