''' All of the utils for GSM8K '''

from __future__ import annotations
import json
import re
import os
import io
import random
from typing import Literal, Tuple, Optional
from datasets import load_dataset
from enum import Enum
import logging

from agents.gsm8k import GSM8KProblem

# parsing constants
ANS_RE = re.compile(r"####\s*\$?\s*([-+]?\d+(?:,\d{3})*(?:\.\d+)?)", re.IGNORECASE)
INVALID_ANS = "[invalid]"

class HFDatasetNames(Enum):
    ''' Containers for which dataset to download '''
    GSMK8 = {
        'main': 'openai/gsm8k',
        'socratic': 'openai/gsm8k'
    }

def read_jsonl_dataset(path: str) -> list[dict[str, str]]:
    """
    Reads jsonl and returns it 
    """
    with open(path) as fh:
        return [json.loads(line) for line in fh.readlines() if line]
    
def get_examples(split):
    path = os.path.join("data/", f"{split}.jsonl")
    examples = read_jsonl_dataset(path)

    for ex in examples:
        ex.update(question=ex["question"] + "\n")
        ex.update(answer=ex["answer"] + "<|endoftext|>")

    print(f"{len(examples)} {split} examples")
    return examples
    
def gsm_extract_answer(completion: str) -> str:
    """ 
    Parses through a string and returns the answer as a str
    
    Expects the answer in this format: 
    Answer is #### -567.89 or #### -567.89. ===> -567.89
    """
    match = ANS_RE.search(completion)
    if match:
        match_str = match.group(1).strip()
        match_str = match_str.replace(",", "")
        match_str = match_str.rstrip('.')
        return match_str
    else:
        return INVALID_ANS
    
def filter_output_type(llm_output: str) -> Literal['question', 'answer', 'final_answer', '[invalid]']:
    """Filter an LLM output and return what kind of response it is."""
    # Patterns
    FA = re.compile(r"####\s*\$?\s*[-+]?\d+(?:,\d{3})*(?:\.\d+)?", re.IGNORECASE)
    Q = re.compile(r"\bQuestion\b", re.IGNORECASE)
    A = re.compile(r"\bAnswer\b", re.IGNORECASE)

    # Search for patterns
    FA_searched = FA.search(llm_output)
    Q_searched = Q.search(llm_output)
    A_searched = A.search(llm_output)

    # Determine the output type
    if FA_searched:
        return 'final_answer'
    elif Q_searched:
        return 'question'
    elif A_searched:
        return 'answer'
    else:
        return '[invalid]'

def gsm_is_correct(idx: int, answer: str, gold_answer: dict[str, str]) -> Tuple[bool, str]:
    """ Checks if final model's output matches the gold answer """ 
    answer = float(gsm_extract_answer(answer))
    gold_answer = float(gsm_extract_answer(gold_answer["answer"]))
    
    return (bool(answer == gold_answer), 
            f'Question #{idx + 1} << Model Guess: {answer} ||| Gold Answer: {gold_answer} >>\n')

# mine 
def batch_sample_gsm(dataset: list[dict[str, str]], batch_size: int) -> list[dict[str, str]]: 
    """ 
    Sample batches of dicts from gsm8k dataset
    [{question: ..., answer: ...}, ...]
    """
    indices: list[int] = random.sample(range(len(dataset)), batch_size)
    sampled_data = [dataset[i] for i in indices]
    
    return sampled_data

def batch_gsm_extract_answer(completions: list[str]) -> list[str]: 
    """ Batch extracts answers """
    answers = list(map(gsm_extract_answer, completions))
    return answers

def batch_eval_gsm(parsed_model_answers: list[str], gsm_qa_pairs: list[dict[str, str]]) -> list[bool]: 
    """ Batch evals model answers to qa pairs from dataset"""
    return [gsm_is_correct(answer, example) for answer, example 
            in zip(parsed_model_answers, gsm_qa_pairs)]

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
                            batch_size: int, 
                            save_dir: str) -> list[str]: 
    """
    Splits GSM8K Dataset Into Chunks and Saves Them Into a Directory.
    """
    datasets = split_dataset(dataset, num_chunks)
    
    # Create the save directory if it doesn't exist
    if not os.path.exists(save_dir):
        print(f'Dataset Chunk Directory Does Not Exist at {save_dir}, Making it...'.upper()) 
        os.makedirs(save_dir)
    
    saved_dataset_paths: list[str] = []
    for idx, dataset_chunk in enumerate(datasets): 
        save_path = os.path.join(save_dir, f'chunk-{idx}-GSM8K-len-{len(dataset_chunk)}.jsonl')  # Proper file naming
        with io.open(save_path, 'w', buffering=4096) as file: 
            file.writelines(json.dumps(entry) + "\n" for entry in dataset_chunk)
        saved_dataset_paths.append(save_path)
    
    return saved_dataset_paths
            
if __name__=="__main__": 
    
    chunk = read_jsonl_dataset('/lus/eagle/projects/FoundEpidem/bhsu/2024_research/agents/agents/data/dataset_chunks/chunk_0_GSM8K.jsonl')
    
    breakpoint()
    
    
    download_datasets(HFDatasetNames)