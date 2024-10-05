from __future__ import annotations

''' Dataset Class for GSM8K '''

import os, json
from typing import Any, Union
from pathlib import Path
from torch.utils.data import Dataset
from datasets import Dataset as HFDataset, load_from_disk

PathLike = Union[str, Path]

def read_jsonl(path: str):
    with open(path) as fh:
        return [json.loads(line) for line in fh.readlines() if line]

def get_examples(): 
    ...

class GSM8KDataset(Dataset): 
    ''' Dataset for GSM8K Datasets '''
    
    def __init__(self, dataset_path: str = '') -> None:
        super().__init__()
        
        dataset: HFDataset = load_from_disk(dataset_path) 
        
        self.dataset = dataset
        
    
    def initialize(self, path: PathLike='') -> None: 
        
        assert os.path.exists(path)
        ... 
        
    def preprocess(self, dataset: HFDataset) -> HFDataset: 
        ...       
        
    def __getitem__(self, index: int) -> Any:
        return self.dataset[index]
    
    def __len__(self) -> int:
        return len(self.dataset)
    
if __name__ == "__main__": 
    
    dataset = GSM8KDataset('./')
    
    breakpoint()