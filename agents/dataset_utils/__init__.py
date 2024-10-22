from __future__ import annotations

''' Downloads for Huggingface Datasets '''

from enum import Enum
from datasets import load_dataset

class Dataset_Info(Enum):
    ''' Containers for which dataset to download '''
    GSMK8 = {
        'main': 'openai/gsm8k',
        'socratic': 'openai/gsm8k'
    }
    
    
