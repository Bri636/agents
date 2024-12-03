""" Main Script for downloading the datasets and running evaluations """
from __future__ import annotations

from agents.utils import BaseConfig

class DatasetConfig(BaseConfig):
    dataset_name_or_path: str = ''
    

class GSMConfig(BaseConfig): 
    ...