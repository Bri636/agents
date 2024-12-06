""" Script for Fragmenting Datasets """

from __future__ import annotations
from typing import Optional, Union
from enum import Enum
from datasets import Dataset as HFDataset, load_dataset
import os, io, json
from argparse import ArgumentParser
from dataclasses import dataclass
import random, logging
from pathlib import Path

import pkg_resources
from agents.gsm8k import read_jsonl_dataset, GSM8KProblem
from agents.utils import configure_logger, get_logging_path


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
            download_messages.append(f'Folder does not exist, making one at {dataset_cache_path}...\n')
            os.makedirs(dataset_cache_path)

        if isinstance(info, dict): 
            list(map(lambda item: load_dataset(item[1], item[0], cache_dir=dataset_cache_path), info.items()))
        
    return download_messages 


def split_dataset(dataset: list[GSM8KProblem], num_chunks: int) -> list[list[GSM8KProblem]]:
    """Splits the dataset into equally sized chunks based on the number of chunks."""
    # Calculate the chunk size
    chunk_size = len(dataset) // num_chunks
    # Create the chunks
    chunks = [dataset[i * chunk_size: (i + 1) * chunk_size] for i in range(num_chunks - 1)]
    # Add the remaining data to the last chunk (in case of leftovers)
    chunks.append(dataset[(num_chunks - 1) * chunk_size:])
    return chunks

def chunk_datasets_and_save(dataset: list[dict], 
                            num_chunks: int,
                            save_dir: str, 
                            logger: Optional[logging.Logger] = None
                            ) -> list[str]: 
    """
    Splits GSM8K Dataset Into Chunks and Saves Them Into a Directory. Returns a List of Paths They are Saved at
    """
    datasets = split_dataset(dataset, num_chunks)
    
    # Create the save directory if it doesn't exist
    if not os.path.exists(save_dir):
        message = f'<<< Dataset Chunk Directory Does Not Exist At {save_dir}, Making It... >>>\n'
        if logger: 
            logger.info(message)

        print(message) 
        os.makedirs(save_dir)
    
    saved_dataset_paths: list[str] = []
    for idx, dataset_chunk in enumerate(datasets): 
        save_path = os.path.join(save_dir, f'chunk-{idx}-GSM8K-len-{len(dataset_chunk)}.jsonl')  # Proper file naming
        with io.open(save_path, 'w', buffering=4096) as file: 
            file.writelines(json.dumps(entry) + "\n" for entry in dataset_chunk)
        saved_dataset_paths.append(save_path)
    
    return saved_dataset_paths

def get_dataset_path(env_name: str = 'agents') -> str: 
    """ Returns to gsm.jsonl filepath based on package name """
    return str(pkg_resources.resource_filename(env_name, '') + f'/data/gsm.jsonl')

@dataclass
class DatasetChunkConfig: 
    dataset_name_or_path: str
    dataset_chunks_save_path: str
    num_chunks: int
    max_size_per_chunk: int
    seed: int
    logging: bool 

def parse_args() -> DatasetChunkConfig: 
    
    arg_parser = ArgumentParser()
    arg_parser.add_argument('--dataset_name_or_path', type=str, default='/lus/eagle/projects/FoundEpidem/bhsu/2024_research/agents/agents/data/gsm.jsonl')
    arg_parser.add_argument('--dataset_chunks_save_path', type=str, default='/lus/eagle/projects/FoundEpidem/bhsu/2024_research/agents/agents/data/')
    arg_parser.add_argument('--num_chunks', type=int, default=4)
    arg_parser.add_argument('--max_size_per_chunk', type=int, default=250)
    arg_parser.add_argument('--seed', type=int, default=10)
    arg_parser.add_argument('--logging', action='store_true')
    
    return DatasetChunkConfig(**vars(arg_parser.parse_args()))

def main(): 
    
    args = parse_args()
    random.seed(args.seed)
    
    log_file_path = get_logging_path(
        file_name=f'dataset_chunk_max_size-{args.max_size_per_chunk}-num_chunks-{args.num_chunks}.log')
    logger = configure_logger('info', logging_save_path=log_file_path)

    if not os.path.exists(args.dataset_name_or_path): 
        message = f'<<< GSM Dataset Does Not Exist at {args.dataset_name_or_path}, Pointing to {get_dataset_path()} Instead... >>>\n'
        if args.logging: 
            logger.info(message)

        print(message)
        args.dataset_name_or_path = get_dataset_path()
        
    dataset = read_jsonl_dataset(args.dataset_name_or_path)
    
    max_size = args.max_size_per_chunk * args.num_chunks
    
    args.dataset_chunks_save_path = Path(args.dataset_chunks_save_path) / Path(f'GSM_max_size-{max_size}_num_chunks-{args.num_chunks}')
    
    if len(dataset) > max_size: 
        message = f'<<< Based On Configs - Max Chunk Length: {args.max_size_per_chunk} & Num Chunks: {args.num_chunks}, Truncating Datatset to Length: {max_size}... >>>\n'
        if logging: 
            logger.info(message)

        print(message)
        dataset = random.sample(dataset, max_size)
    
    save_paths = chunk_datasets_and_save(dataset, 
                                         args.num_chunks, 
                                         args.dataset_chunks_save_path, 
                                         logger)

    for path in save_paths: 
        message = f'<<< Saved {args.num_chunks} Dataset Chunks of Size {args.max_size_per_chunk} at Path: {path} >>>\n'
        if args.logging: 
            logger.info(message)

        print(message)

if __name__=="__main__": 
    
    main()