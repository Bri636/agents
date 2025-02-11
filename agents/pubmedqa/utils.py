""" Utils for reading and processing PubMedQA Data """

from __future__ import annotations
from datasets import load_dataset, Dataset
from typing import TypedDict, Optional, Literal, Tuple
import logging
import re
from rich.console import Console
from rich.panel import Panel
from rich.text import Text
import time, random

# parsing constants
# ANS_RE = re.compile(r"####\s*\$?\s*([-+]?\d+(?:,\d{3})*(?:\.\d+)?)", re.IGNORECASE)
ANS_RE = re.compile(r"####\s*\$?\s*([-+]?\d+(?:,\d{3})*(?:\.\d+)?|Yes|No|yes|no)", re.IGNORECASE)
INVALID_ANS = "[invalid]"

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
    
def pubmed_extract_answer(completion: str) -> str:
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
    """ Filter an LLM output and return what kind of response it is."""
    # Patterns
    # FA = re.compile(r"####\s*\$?\s*[-+]?\d+(?:,\d{3})*(?:\.\d+)?", re.IGNORECASE)
    FA = re.compile(r"####\s*\$?\s*([-+]?\d+(?:,\d{3})*(?:\.\d+)?|Yes|No)", re.IGNORECASE)
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
    

def question_is_correct(idx: int, answer: str, gold_answer: dict[str, str]) -> Tuple[bool, str]:
    """ Checks if final model's output matches the gold answer """ 
    answer = pubmed_extract_answer(answer).lower()
    gold_answer = pubmed_extract_answer('####' + gold_answer["final_decision"]).lower()
    return (bool(answer == gold_answer), 
            f'Question #{idx + 1} << Model Guess: {answer} ||| Gold Answer: {gold_answer} >>\n')

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
    
    
##### visual utils ####
def print_evaluation_start(console: Console, strategy: str):
    console.rule(f'Running Eval on {strategy} Reasoner', style="bold", characters='=')
    for _ in range(2):
        console.rule(f'', style="bold", characters='=')
    time.sleep(1)
    
def print_batch_progress(console: Console, batch_idx: int, num_batches: int, panels: list, messages: list, num_correct: int, batch_size: int):
    console.rule(f"Batch {batch_idx + 1} of {num_batches} Total Batches", style="bold", characters='=')
    if all(panels):
        idx_to_show = random.choice(range(len(messages)))
        console.print(panels[idx_to_show])

    reasoner_correct_text = Text.assemble(
        ("* Reasoner: ", "bold red"),
        (f"Sample {messages[idx_to_show]}\n", "white"),
        ("* Correct: ", "bold red"),
        (f"{num_correct} Questions Correct Out of {int((batch_idx + 1) * batch_size)} Total Questions Asked... Score: {((num_correct / int((batch_idx + 1) * batch_size)) * 100):.2f} %", "white")
    )
    panel_summ = Panel(
        reasoner_correct_text,
        border_style="white",
        title="Status",
        title_align="center",
        expand=True
    )
    console.print(panel_summ)