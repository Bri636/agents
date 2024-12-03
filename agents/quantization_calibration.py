""" Quantizes your model of choice with W8A8 Quantization """

from llmcompressor.transformers import SparseAutoModelForCausalLM
from transformers import AutoTokenizer
from llmcompressor.transformers import oneshot
from llmcompressor.modifiers.quantization import GPTQModifier
from llmcompressor.modifiers.smoothquant import SmoothQuantModifier
from datasets import load_dataset
from argparse import ArgumentParser, Namespace
from dataclasses import dataclass

MODELS = {
    'llama': 'meta-llama/Meta-Llama-3-8B-Instruct',
    'mistral': 'mistralai/Ministral-8B-Instruct-2410', 
    'google': 'google/gemma-2-2b-it'
}

@dataclass
class QuantizationConfig: 
    """ Simple container so we can see the Namespace """
    model_name_or_path: str
    huggingface_dataset_path_or_name: str
    save_dir: str
    max_sequence_length: int 
    num_calibration_samples: int 
    seed: int
    
    def __post_init__(self):
        """ Makes the save dir from the trunc name"""
        self.save_dir = str(str(self.save_dir) + self.model_name_or_path.split("/")[1] + "-W8A8-Dynamic-Per-Token")
        
def parse_arguments() -> QuantizationConfig: 
    """ Parses some arguments used for quantization calibration """
    argparser = ArgumentParser()
    argparser.add_argument('--model_name_or_path', type=str, default='meta-llama/Meta-Llama-3-8B-Instruct')
    argparser.add_argument('--huggingface_dataset_path_or_name', type=str, default="HuggingFaceH4/ultrachat_200k")
    argparser.add_argument('--save_dir', type=str, default='./')
    argparser.add_argument('--max_sequence_length', type=int, default=2048)
    argparser.add_argument('--num_calibration_samples', type=int, default=512)
    argparser.add_argument('--seed', type=int, default=10)
    args_dict = vars(argparser.parse_args())

    return QuantizationConfig(**args_dict)

def main():
    
    args = parse_arguments()
    
    # model = SparseAutoModelForCausalLM.from_pretrained(
    # args.model_name_or_path, device_map="auto", torch_dtype="auto",
    # )
    model = SparseAutoModelForCausalLM.from_pretrained(
    args.model_name_or_path, device_map="auto", torch_dtype="auto",
    )
    
    tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path)
    # Configure the quantization algorithms
    recipe = [
        SmoothQuantModifier(smoothing_strength=0.8),
        GPTQModifier(targets="Linear", scheme="W8A8", ignore=["lm_head"]),
    ]
    # Load and preprocess the dataset
    ds = load_dataset(args.huggingface_dataset_path_or_name, split="train_sft")
    ds = ds.shuffle(seed=args.seed).select(range(args.num_calibration_samples))

    def preprocess(example):
        return {"text": tokenizer.apply_chat_template(example["messages"], tokenize=False)}
    ds = ds.map(preprocess)

    def tokenize(sample):
        return tokenizer(sample["text"], 
                         padding=False, 
                         max_length=args.max_sequence_length, 
                         truncation=True, 
                         add_special_tokens=False)
    ds = ds.map(tokenize, remove_columns=ds.column_names)

    # Apply quantization
    oneshot(
        model=model,
        dataset=ds,
        recipe=recipe,
        max_seq_length=args.max_sequence_length,
        num_calibration_samples=args.num_calibration_samples,
    )

    # Save the compressed model
    model.save_pretrained(args.save_dir, save_compressed=True)
    tokenizer.save_pretrained(args.save_dir)
    
if __name__=="__main__": 
    
    main()