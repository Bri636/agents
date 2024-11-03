import json
import re
import os
import random
from typing import Literal

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

ANS_RE = re.compile(r"#### (\-?[0-9\.\,]+)")
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
    Answer is #### -567.89 ===> -567.89
    """
    
    match = ANS_RE.search(completion)
    if match:
        match_str = match.group(1).strip()
        match_str = match_str.replace(",", "")
        return match_str
    else:
        return INVALID_ANS
    
# def gsm_is_correct(model_completion: str, gt_example: dict[str, str]) -> bool:
#     gt_answer = gsm_extract_answer(gt_example["answer"])
#     assert gt_answer != INVALID_ANS
#     return gsm_extract_answer(model_completion) == gt_answer

def gsm_is_correct(answer: str, gold_answer: dict[str, str]) -> bool:
    """ Checks if final model's output matches the gold answer """ 
    return bool(float(gsm_extract_answer(answer)) == 
                float(gsm_extract_answer(gold_answer["answer"])))

# mine 
def batch_sample_qa_pairs(dataset: list[dict[str, str]], batch_size: int) -> list[dict[str, str]]: 
    """ Sample batches from gsm8k dataset """
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
    
def filter_output_type(llm_output: str) -> Literal['question', 'answer', 'final_answer', '[invalid]']: 
    """ Filter an llm output and returns what kind of response it is """
    Q = re.compile(r"Question (\-?[0-9\.\,]+)")
    A = re.compile(r"Answer (\-?[0-9\.\,]+)")
    FA = re.compile(r"####")
    
    if Q.search(llm_output): 
        return 'question'
    elif A.search(llm_output): 
        return 'answer'
    elif FA.search(llm_output): 
        return 'final_answer'
    else: 
        return '[invalid]'