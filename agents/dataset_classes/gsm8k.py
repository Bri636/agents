from __future__ import annotations

''' Dataset Class for GSM8K: https://github.com/openai/grade-school-math '''

import os, json
from typing import Any, Union
from pathlib import Path
from torch.utils.data import Dataset
import torch
import re
from transformers import AutoTokenizer

from agents.configs import BaseConfig

PathLike = Union[str, Path]
ANS_RE = re.compile(r"#### (\-?[0-9\.\,]+)")
INVALID_ANS = "[invalid]"

class GSM8KConfig(BaseConfig): 
    ''' Config for GSM8K Dataset Loading '''
    ...

def read_jsonl(path: str):
    with open(path) as fh:
        return [json.loads(line) for line in fh.readlines() if line]

def get_examples(split: str):
    path = os.path.join("data/", f"{split}.jsonl")
    examples = read_jsonl(path)

    for ex in examples:
        ex.update(question=ex["question"] + "\n")
        ex.update(answer=ex["answer"] + "<|endoftext|>")

    print(f"{len(examples)} {split} examples")
    return examples

def extract_answer(completion):
    match = ANS_RE.search(completion)
    if match:
        match_str = match.group(1).strip()
        match_str = match_str.replace(",", "")
        return match_str
    else:
        return INVALID_ANS

def is_correct(model_completion, gt_example):
    gt_answer = extract_answer(gt_example["answer"])
    assert gt_answer != INVALID_ANS
    return extract_answer(model_completion) == gt_answer

class GSMDataset(Dataset):
    def __init__(self, tokenizer: AutoTokenizer, examples: list, loss_on_prefix=True):
        self.examples = examples
        self.qns = [ex["question"] for ex in self.examples]
        self.ans = [ex["answer"] for ex in self.examples]
        self.qns = tokenizer(self.qns, padding=False)
        self.ans = tokenizer(self.ans, padding=False)
        self.loss_on_prefix = loss_on_prefix
        self.max_len = max(
            [
                len(self.qns["input_ids"][i]) + len(self.ans["input_ids"][i])
                for i in range(len(self.examples))
            ]
        )
        print(f"Max tokens: {self.max_len}")

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx):
        qn_tokens = self.qns["input_ids"][idx]
        ans_tokens = self.ans["input_ids"][idx]
        pad_tokens = [0] * (self.max_len - len(qn_tokens) - len(ans_tokens))
        tokens = qn_tokens + ans_tokens + pad_tokens
        mask = (
            ([int(self.loss_on_prefix)] * len(qn_tokens))
            + ([1] * len(ans_tokens))
            + ([0] * len(pad_tokens))
        )
        tokens = torch.tensor(tokens)
        mask = torch.tensor(mask)
        return dict(input_ids=tokens, attention_mask=mask)
    
if __name__ == "__main__": 
    
    tokenizer = AutoTokenizer.from_pretrained('meta-llama/Meta-Llama-3-70B')
    
    train_examples = get_examples("train")
    test_examples = get_examples("test")
    breakpoint()
    
    dataset = GSMDataset(tokenizer, train_examples, True)
    
    breakpoint()