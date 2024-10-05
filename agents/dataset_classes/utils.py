from __future__ import annotations

'''Utils for loading datasets'''

from datasets import load_dataset
import os
from agents.dataset_classes import Dataset_Info

def download_datasets(dataset_info: Dataset_Info, 
                      cache_dir: str = '/Users/BrianHsu/Desktop/GitHub/agents/agents/data/'
                      ) -> list[str]: 
    ''' Downloads datasets '''
    
    download_messages=[]
    for dataset in dataset_info: 
        info = dataset.value
        folder_name = list(info.values())[0] if isinstance(info, dict) else info
        dataset_cache_path = cache_dir + f'{folder_name}'
        
        if not os.path.exists(dataset_cache_path): 
            download_messages.append(f'Folder does not exist, making one at {dataset_cache_path}...')
            os.makedirs(dataset_cache_path)

        if isinstance(info, dict): 
            list(map(lambda item: load_dataset(item[1], item[0], cache_dir=dataset_cache_path), info.items()))
        
    return download_messages    
            
if __name__=="__main__": 
    
    download_datasets(Dataset_Info)
    