""" Evaluator class for evaluating how well the llm agents perform """

from __future__ import annotations
from typing import Callable, Tuple, Any, Optional, Literal, Union
import random
import time
from tqdm import tqdm
from tqdm.rich import tqdm
from dataclasses import dataclass, asdict
from rich.console import Console
from rich.panel import Panel
from rich.text import Text
import logging

from agents.gsm8k.utils import read_jsonl_dataset, filter_output_type, gsm_is_correct
from agents.generators.base_generator import BaseLLMGenerator
from agents.generators.vllm_generator import VLLMGenerator, VLLMGeneratorConfig
from agents.prompts.llama_prompt import GSMLlamaPromptTemplate
from agents.prompts.standard_prompt import StandardGSMPromptTemplate
from agents.reasoners.base_reasoner import BaseReasoner
from agents.reasoners.reasoner import LLMReasoner
from agents.reasoners.wm_reasoner import WorldReasoner
from agents.reasoners.wm_mcts_reasoner import MCTSWorldReasoner
from agents.utils import BaseConfig, register_strategy
from agents.callbacks import ThroughputCallback, ThroughputMetrics, BatchMetrics
# import types
from agents.gsm8k import T
from agents.callbacks import Callback, CallbackMetrics, Registered_Callbacks

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
 
@dataclass
class GSMEvaluationConfig:
    """
    Configuration class for GSM evaluation.

    This class defines the settings and parameters used during the evaluation 
    process of GSM (Generalized Small Models). The parameters control various 
    aspects of the evaluation, such as the dataset path, randomization, 
    verbosity, and batch processing.

    Attributes:
        dataset_path (str): 
            The path to the dataset used for evaluation. This is a required 
            field and should point to the appropriate dataset file or directory.

        seed (int, optional): 
            The random seed for ensuring reproducibility of evaluation results. 
            Defaults to 10.

        disable_tqdm (bool, optional): 
            A flag to disable the TQDM progress bar during evaluation. 
            Set to `True` to disable TQDM (useful for non-interactive 
            environments). Defaults to `True`.

        num_samples (int, optional): 
            The total number of samples to evaluate. Defaults to 7456.

        num_tries (int, optional): 
            The number of attempts to make for each sample during evaluation. 
            This allows for multiple evaluation attempts for improved robustness. 
            Defaults to 10.

        batch_size (int, optional): 
            The number of samples to process in a single batch during evaluation. 
            This controls memory usage and can be adjusted based on hardware 
            capabilities. Defaults to 32.
    """
    dataset_path: str = '/lus/eagle/projects/FoundEpidem/bhsu/2024_research/agents/agents/data/gsm.jsonl'
    seed: int = 10
    disable_tqdm: bool = True
    num_samples: int = 7456
    num_tries: int = 10
    batch_size: int = 32
    strategy: str = 'mcts_world_model'
    
 
def batch_gsm_evaluate(
    strategy: str,
    dataset: list[dict[str, str]],
    reasoner: BaseReasoner,
    seed: int = 10,
    disable_tqdm: bool = True,
    num_samples: int = 100,
    batch_size: int = 32,
    num_tries: int = 10, 
    callbacks: Optional[list[Callback] | Callback] = None,
    logger: Optional[logging.Logger] = None
) -> Tuple[Metrics, list[CallbackMetrics]]:
    """ Performs batched evaluation on N samples from GSM8K within M tries, and calculates the metrics for them """
    from agents.utils import batch_data_with_indices
    from agents.gsm8k.types import GSM8KProblem
    from agents.callbacks.callbacks import ThroughputCallback, ThroughputMetrics

    # setting up and double checking callback stuff
    assert num_samples % batch_size == 0, f'''
Num_samples is not divisible by batch_size !'''
    random.seed(seed)
    
    if not isinstance(callbacks, list): 
        callbacks = [callbacks]
    
    # for if batch_size is larger than num_samples
    num_samples = max(num_samples, batch_size)
    sample_indices = random.sample(range(len(dataset)), num_samples)
    samples = [dataset[i] for i in sample_indices]
    num_batches = int(num_samples / batch_size)

    batched_samples, batch_indices = batch_data_with_indices(samples,
                                                       sample_indices,
                                                       batch_size)

    console = Console()
    console.rule(
        f'Running Eval on {strategy} Reasoner', style="bold", characters='=')
    map(lambda _: console.rule(f'', style="bold", characters='='), range(2))
    time.sleep(1)

    [callback.on_start() for callback in callbacks] # turn on all callbacks
    
    num_correct = 0
    num_batches_completed = 0
    with tqdm(total=num_samples, disable=disable_tqdm, desc=f"GSM Evaluation - {num_samples} Samples", leave=False) as progress_bar:

        for batch_idx, (batch, indices) in enumerate(zip(batched_samples, batch_indices)):
            # callbacks on batch start
            [callback.on_batch_start() for callback in callbacks]
            
            finished, corrects, messages, panels = reasoner.batch_generate_answer(indices, batch, num_tries)
            if finished: 
                num_correct += sum(corrects)
                num_batches_completed += 1  # Fixed increment
            reasoner.reset_pass() # reset prompts
            # Printing results for each sample
            idx_to_show = random.sample(range(len(messages)), 1)[0] # random sample from batch for a panel 
            large_bar = f"Batch {batch_idx + 1} of {num_batches} Total Batches"
            console.rule(large_bar, style="bold", characters='=')
            if all(panels):
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
            # logging statistics
            [callback.on_batch_end(batch_idx=batch_idx, batch_size=batch_size) 
             for callback in callbacks]
            if logger: 
                logger.info(f"{num_correct} Questions Correct Out of {int((batch_idx + 1) * batch_size)} Total Questions Asked... Score: {((num_correct / int((batch_idx + 1) * batch_size)) * 100):.2f} %\n")
            else:
                print(f"{num_correct} Questions Correct Out of {int((batch_idx + 1) * batch_size)} Total Questions Asked... Score: {((num_correct / int((batch_idx + 1) * batch_size)) * 100):.2f} %\n")
            console.print(panel_summ)
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
                      }), 
            callback_metrics
            )

def gsm_evaluate(
    strategy: str,
    dataset: list[dict[str, str]],
    reasoner: BaseReasoner,
    seed: int = 10,
    disable_tqdm: bool = True,
    num_samples: int = 100,
    num_tries: int = 10
) -> Metrics:
    """ Performs evaluation on N samples from GSM8K within M tries, and calculates the metrics for them """
    random.seed(seed)
    sample_idx = random.sample(range(len(dataset)), num_samples)
    samples = [dataset[i] for i in sample_idx]

    console = Console()
    console.rule(
        f'Running Eval on {strategy} Reasoner', style="bold", characters='=')
    map(lambda _: console.rule(f'', style="bold", characters='='), range(2))
    time.sleep(1)

    num_correct = 0
    num_completed = 0
    with tqdm(total=num_samples, disable=disable_tqdm, desc=f"GSM Evaluation - {num_samples} Samples", leave=False) as progress_bar:
        for idx, sample in enumerate(samples):
            finished, correct, message, panel = reasoner.generate_answer(
                idx, sample, num_tries)
            num_correct += finished and correct
            num_completed += finished
            reasoner.reset_pass()
            # Printing results for each sample
            large_bar = f"Question {idx + 1}"
            console.rule(large_bar, style="bold", characters='=')
            if panel:
                console.print(panel)

            reasoner_correct_text = Text.assemble(
                ("* Reasoner: ", "bold red"),
                (f"{message}\n", "white"),
                ("* Correct: ", "bold red"),
                (f"{num_correct} Questions Correct Out of {idx + 1} Total Questions Asked... Score: {((num_correct / (idx +1)) * 100):.2f} %", "white")
            )
            panel_summ = Panel(
                reasoner_correct_text,
                border_style="white",
                title="Status",
                title_align="center",
                expand=True
            )
            console.print(panel_summ)
            progress_bar.update(1)

    percent_completed = (num_completed / num_samples) * 100
    percent_correct = (num_correct / num_completed) * \
        100 if num_completed > 0 else 0.0

    return Metrics(**{'completed': percent_completed, 
                      'correct': percent_correct, 
                      'num_correct': num_correct, 
                      'num_completed': num_completed, 
                      'num_total': num_samples
                      })

