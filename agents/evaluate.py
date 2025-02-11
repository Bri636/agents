""" Evaluator class for evaluating how well the llm agents perform """
from __future__ import annotations
from typing import Callable, Tuple, Any, Optional, Literal, Union
import random, os
from tqdm import tqdm
from tqdm.rich import tqdm
from dataclasses import dataclass, asdict
from rich.console import Console
from datasets import Dataset
from argparse import ArgumentParser
from pydantic import Field
from datasets import load_dataset

from agents.pubmedqa import filter_output_type, print_batch_progress, print_evaluation_start
from agents.utils import BaseConfig, batch_data_with_indices
from agents.callbacks import Callback, CallbackMetrics, ThroughputCallback, GSMThroughputMetrics
from agents.reasoners import BaseReasoner, MCTSWorldReasoner
from agents.generators import VLLMGenerator, VLLMGeneratorConfig
from agents.prompts import PubMedPromptTemplate

@dataclass
class Metrics:
    """
    A dataclass to track progress metrics.

    Attributes:
        percent_completed (float): The percentage of questions that have been completed.
        percent_correct (float): The percentage of completed questions that were answered correctly.
        num_completed (int): The total number of questions that have been completed. This value is always greater than or equal to `num_correct`.
        num_correct (int): The total number of completed questions that were answered correctly.
        num_total (int): The total number of questions or samples that were run for a function.
    """
    percent_completed: float
    """ The percentage of questions that have been completed. """
    percent_correct: float
    """ The percentage of completed questions that were answered correctly. """
    num_completed: int
    """ The total number of questions that have been completed. """
    num_correct: int
    """ The total number of completed questions that were answered correctly. """
    num_total: int
    """ The total number of questions or samples that were run for a function. """

class PubMedEvaluationConfig(BaseConfig):
    """
    Configuration class for PubMed evaluation.

    This class defines the settings and parameters used during the evaluation 
    process of PubMed-related models. The parameters control various 
    aspects of the evaluation, such as the dataset path, randomization, 
    verbosity, and batch processing.
    """
    verbose: bool = Field(default=False)
    """ Whether to print out outputs or not """
    disable_tqdm: bool = Field(default=True)
    "A flag to disable the TQDM progress bar during evaluation. Defaults to `True`."
    num_samples: int = Field(default=1000)
    """The total number of samples to evaluate. Defaults to 1000."""
    num_tries: int = Field(default=10)
    """The number of attempts to make for each sample during evaluation. Defaults to 10."""
    batch_size: int = Field(default=4)
    """The number of samples to process in a single batch during evaluation. Defaults to 4."""
    generator_config: "VLLMGeneratorConfig" = Field(
        default_factory=VLLMGeneratorConfig)
    """Configuration for the VLLM generator."""
    
    def __init__(self, **kwargs):
        """Initialize the evaluation config with proper validation."""
        super().__init__(**kwargs)
        self.num_samples = max(self.num_samples, self.batch_size)
        assert self.num_samples % self.batch_size == 0, \
            "num_samples must be divisible by batch_size!"

def batch_gsm_evaluate(
    strategy: str,
    dataset: Dataset,
    reasoner: BaseReasoner,
    verbose: bool = True,
    disable_tqdm: bool = True,
    num_samples: int = 100,
    batch_size: int = 32,
    num_tries: int = 10,
    callbacks: Optional[list[Callback]] = None
) -> Tuple[Metrics, list[CallbackMetrics]]:
    """ Performs batched evaluation on N samples from GSM8K within M tries, and calculates the metrics for them """
    # generating batches
    sample_indices = random.sample(range(len(dataset)), num_samples)
    samples = [dataset[i] for i in sample_indices]
    num_batches = int(num_samples / batch_size)
    batched_samples, batch_indices = batch_data_with_indices(samples, sample_indices, batch_size)
    console = Console()
    print_evaluation_start(console, strategy) if verbose else None
    # start callbacks 
    if callbacks: 
        [callback.on_start() for callback in callbacks]
        
    # set counters for number of correct questions and number of batches completed
    num_correct = 0
    num_batches_completed = 0
    with tqdm(total=num_samples,
              disable=disable_tqdm,
              desc=f"PubMed Evaluation - {num_samples} Samples",
              leave=False) as progress_bar:

        for batch_idx, (batch, indices) in enumerate(zip(batched_samples, batch_indices)):
            # callbacks on batch start
            if callbacks: 
                [callback.on_batch_start() for callback in callbacks]
                
            finished, corrects, messages, panels = reasoner.batch_generate_answer(indices, batch, num_tries)
            if finished:
                num_correct += sum(corrects)
                num_batches_completed += 1  # Fixed increment
            reasoner.reset_pass()  # reset prompts
            print_batch_progress(console, batch_idx, num_batches, panels, messages, num_correct, batch_size) if verbose else None
            # logging statistics
            # NOTE - is num_batches the same as num_steps?
            if callbacks:
                [callback.on_batch_end(batch_idx=batch_idx, 
                                       batch_size=batch_size, 
                                       num_steps=num_batches)
                 for callback in callbacks]
                batch_metrics: list[dict] = [asdict(callback.return_metrics()) for callback in callbacks]
            
            print(f"""
{num_correct} Questions Correct Out of {int((batch_idx + 1) * batch_size)}\\
Total Questions Asked... Score: {((num_correct / int((batch_idx + 1) * batch_size)) * 100):.2f} %\n""") if verbose else None
            # print(f'Device {os.environ.get("CUDA_VISIBLE_DEVICES")}: Batch Metrics for Batch {batch_idx + 1}: {pp.pformat(batch_metrics)}\n')
            progress_bar.update(1)

    percent_completed = (int(num_batches_completed * batch_size) / num_samples) * 100
    percent_correct = (num_correct / int(num_batches_completed * batch_size)) * \
        100 if num_batches_completed > 0 else 0.0

    if callbacks:
        callback_metrics: list[CallbackMetrics] = [callback.return_metrics()
                                                   for callback in callbacks]
    # TODO: figure out the discrepancy from percent_correct and batch_wise acuracy
    return (Metrics(**{'percent_completed': percent_completed,
                       'percent_correct': percent_correct,
                       'num_correct': num_correct,
                       'num_completed': num_batches_completed,
                       'num_total': num_samples
                       }), callback_metrics)

def parse_args() -> Any:
    arg_parser = ArgumentParser()
    arg_parser.add_argument('--dataset_name_or_path', type=str,
                            default='qiaojin/PubMedQA')
    arg_parser.add_argument('--model_name_or_path', type=str,
                            default='meta-llama/Meta-Llama-3-8B-Instruct')
    arg_parser.add_argument('--strategy', type=str,
                            default='mcts_world_model')
    arg_parser.add_argument('--evaluation_config_path', type=str,
                            default='/lus/eagle/projects/FoundEpidem/bhsu/2024_research/agents/agents/config_files/pubmed_eval.yaml')
    arg_parser.add_argument('--logging_save_path', type=str,
                            default='./logs/mcts_world_model.log')
    return arg_parser.parse_args()

if __name__ == "__main__":

    args = parse_args()
    eval_config = PubMedEvaluationConfig.from_yaml(args.evaluation_config_path)
    dataset = load_dataset(args.dataset_name_or_path, 'pqa_labeled',
                           trust_remote_code=True)['train']
    generator = VLLMGenerator(args.model_name_or_path,
                              eval_config.generator_config)
    
    reasoner_registry = BaseReasoner.get_registery()
    reasoner_cls = reasoner_registry[args.strategy]
    
    question_prompt, answer_prompt = PubMedPromptTemplate('question', 2), PubMedPromptTemplate('answer', 2)
    reasoner = reasoner_cls(generator, question_prompt, answer_prompt, filter_output_type)

    output = batch_gsm_evaluate(strategy=args.strategy,
                                dataset=dataset,
                                reasoner=reasoner,
                                disable_tqdm=eval_config.disable_tqdm,
                                num_samples=eval_config.num_samples,
                                batch_size=eval_config.batch_size,
                                num_tries=eval_config.num_tries,
                                callbacks=None)
    breakpoint()
