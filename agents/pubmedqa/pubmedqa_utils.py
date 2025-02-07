""" Utils for reading and processing PubMedQA Data """

from __future__ import annotations
from datasets import load_dataset, Dataset
from typing import TypedDict, Optional
import logging

class PubMedContext(TypedDict): 
    contexts: list[str]
    labels: list[str]
    meshes: list[str]
    reasoning_required_pred: list[str]
    reasoning_free_pred: list[str]

class PubMedProblem(TypedDict): 
    pubid: int
    question: str 
    context: PubMedContext
    long_answer: str
    final_decision: str

def truncate_dataset(dataset: list[PubMedProblem], 
                     batch_size: int, 
                     logger: Optional[logging.Logger]=None) -> list[PubMedProblem]:
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
    
def split_dataset(dataset: list[PubMedProblem], num_chunks: int, batch_size: int) -> list[list[PubMedProblem]]:
    """Chunks dataset into equally sized chunks divisible by batch size."""
    # Calculate the total number of samples to use (divisible by batch_size * num_chunks)
    total_samples = (len(dataset) // (batch_size * num_chunks)) * (batch_size * num_chunks)
    # Adjust the dataset to use only the total_samples
    dataset = dataset[:total_samples]
    # Calculate the chunk size (equal for all chunks and divisible by batch_size)
    chunk_size = total_samples // num_chunks

    return [dataset[i * chunk_size: (i + 1) * chunk_size] 
            for i in range(num_chunks)]    

if __name__=="__main__": 
    
    
    dataset = load_dataset('qiaojin/PubMedQA', 'pqa_labeled')
    
    breakpoint()