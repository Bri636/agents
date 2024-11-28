import json
import re
import os
import random
from typing import Literal, Tuple

"""
Format: 
[{'question': // text question //, 
    'answer': // Step-by-step answer, with "\ n" representing steps of thought, and ####space ... as the answer: 
}]

ex. 
[{'question': 
A cleaning company produces two sanitizer sprays. 
One spray kills 50% of germs, and another spray kills 25% of germs. 
However, 5% of the germs they kill are the same ones. 
What percentage of germs would be left after using both sanitizer sprays together?
    'answer': 
After the first spray kills 50% of germs, there will be 100 - 50 = <<100-50=50>>50% left.
The second spray kills 25%, but 5% have already been killed by the 50% spray, so it kills 25 - 5 = <<25-5=20>>20%.
After the second spray kills 20% of the remaining germs, there will be 50 - 20 = <<50-20=30>>30% left.
#### 30
}]

"""

ANS_RE = re.compile(r"####\s*\$?\s*([-+]?\d+(?:,\d{3})*(?:\.\d+)?)", re.IGNORECASE)
INVALID_ANS = "[invalid]"

def read_jsonl(path: str) -> list[dict[str, str]]:
    """
    Reads jsonl and returns it 
    """
    with open(path) as fh:
        return [json.loads(line) for line in fh.readlines() if line]
    
def get_examples(split):
    path = os.path.join("data/", f"{split}.jsonl")
    examples = read_jsonl(path)

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
    
# def gsm_is_correct(model_completion: str, gt_example: dict[str, str]) -> bool:
#     gt_answer = gsm_extract_answer(gt_example["answer"])
#     assert gt_answer != INVALID_ANS
#     return gsm_extract_answer(model_completion) == gt_answer

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

    