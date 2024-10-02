from __future__ import annotations

''' Dataset Class for GSM8K '''

import os
from typing import Any
from torch.utils.data import Dataset
from datasets import Dataset as HFDataset, load_from_disk

class GSM8KDataset(Dataset): 
    ''' Dataset for GSM8K Datasets '''
    
    def __init__(self, dataset_path: str) -> None:
        super().__init__()
        
        dataset: HFDataset = load_from_disk(dataset_path) 
        
        self.dataset = dataset
        
    def preprocess(self, dataset: HFDataset) -> HFDataset: 
        
        ...       
        
    def __getitem__(self, index: int) -> Any:
        return self.dataset[index]
    
    def __len__(self) -> int:
        return len(self.dataset)
    
    
if __name__ == "__main__": 
    
    dataset = GSM8KDataset()
    
    breakpoint()