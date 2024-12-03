''' Utils for loading datasets'''

from __future__ import annotations
from datasets import load_dataset
import os
from enum import Enum

class HFDatasetNames(Enum):
    ''' Containers for which dataset to download '''
    GSMK8 = {
        'main': 'openai/gsm8k',
        'socratic': 'openai/gsm8k'
    }

def download_datasets(dataset_info: HFDatasetNames, 
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

def truncate_dataset(dataset: list[GSM8KProblem], 
                     batch_size: int, 
                     logger: Optional[logging.Logger]=None) -> list[GSM8KProblem]:
    """
    Truncates the dataset to the largest size divisible by the batch size.

    Args:
        dataset (List[T]): The input dataset to truncate.
        batch_size (int): The batch size to make the dataset divisible by.

    Returns:
        List[T]: A truncated dataset with size divisible by batch_size.
    """
    # Calculate the largest size divisible by the batch size
    truncated_size = (len(dataset) // batch_size) * batch_size
    message = f"Dataset Length {len(dataset)} Not Divisible by Batch Size: {batch_size}, truncating to {truncated_size}..."
    if logger: 
        logger.info(message)
    else: 
        print(message)
    return dataset[:truncated_size] 


def chunk_datasets_and_save(dataset_name_or_path: str, batch_size: int, save_dir: str) -> None: 
    
    ...  
            
if __name__=="__main__": 
    
    download_datasets(HFDatasetNames)
    