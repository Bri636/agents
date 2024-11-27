""" Evaluator class for evaluating how well the llm agents perform """

from __future__ import annotations
from typing import Callable, Tuple, Any
import random
import time
from tqdm import tqdm
from tqdm.rich import tqdm
from dataclasses import dataclass, asdict
from rich.console import Console
from rich.panel import Panel
from rich.text import Text

from agents.gsm8k.utils import read_jsonl, filter_output_type, gsm_is_correct
from agents.generators.base_generator import BaseLLMGenerator
from agents.generators.vllm_generator import VLLMGenerator, VLLMGeneratorConfig
from agents.prompts.llama_prompt import GSMLlamaPromptTemplate
from agents.prompts.standard_prompt import StandardGSMPromptTemplate
from agents.reasoners.base_reasoner import BaseReasoner
from agents.reasoners.reasoner import LLMReasoner
from agents.reasoners.wm_reasoner import WorldReasoner
from agents.utils import BaseConfig, register_strategy
# import types
from agents.gsm8k import T


@dataclass
class Metrics:
    """
    A dataclass to track progress metrics.

    Attributes:
        completed (float): Percent of questions completed.
        correct (float): Percent of completed questions correct.
    """
    completed: float
    """ Percent of questions completed """
    correct: float
    """ Percent of completed questions correct """


@dataclass
class GSMEvaluationConfig:
    """ Config for GSM evaluation """
    dataset_path: str = '/lus/eagle/projects/FoundEpidem/bhsu/2024_research/agents/agents/data/gsm.jsonl'
    seed: int = 10
    disable_tqdm: bool = True
    num_samples: int = 7456
    num_tries: int = 10
    batch_size: int = 32
 # 64: 7488

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

    return Metrics(**{'completed': percent_completed, 'correct': percent_correct})


def batch_gsm_evaluate(
    strategy: str,
    dataset: list[dict[str, str]],
    reasoner: BaseReasoner,
    seed: int = 10,
    disable_tqdm: bool = True,
    num_samples: int = 100,
    batch_size: int = 32,
    num_tries: int = 10
) -> Metrics:
    """ Performs batched evaluation on N samples from GSM8K within M tries, and calculates the metrics for them """
    from agents.utils import batch_data_with_indices
    from agents.gsm8k.types import GSM8KProblem
    
    assert num_samples % batch_size == 0, f'''
Num_samples is not divisible by batch_size !'''

    random.seed(seed)
    # for if batch_size is larger than num_samples
    num_samples = max(num_samples, batch_size)
    sample_indices = random.sample(range(len(dataset)), num_samples)
    samples = [dataset[i] for i in sample_indices]

    batched_samples, batch_indices = batch_data_with_indices(samples,
                                                       sample_indices,
                                                       batch_size)

    console = Console()
    console.rule(
        f'Running Eval on {strategy} Reasoner', style="bold", characters='=')
    map(lambda _: console.rule(f'', style="bold", characters='='), range(2))
    time.sleep(1)

    num_correct = 0
    num_batches_completed = 0
    with tqdm(total=num_samples, disable=disable_tqdm, desc=f"GSM Evaluation - {num_samples} Samples", leave=False) as progress_bar:

        for batch_idx, (batch, indices) in enumerate(zip(batched_samples, batch_indices)):
            finished, corrects, messages, panels = reasoner.batch_generate_answer(indices, batch, num_tries)
            if finished: 
                num_correct += sum(corrects)
                num_batches_completed += 1  # Fixed increment
            reasoner.reset_pass() # reset prompts
            # Printing results for each sample
            idx_to_show = random.sample(range(len(messages)), 1)[0] # random sample from batch for a panel 
            large_bar = f"Question {batch_idx + 1}"
            console.rule(large_bar, style="bold", characters='=')
            if all(panels):
                console.print(panels[idx_to_show])

            reasoner_correct_text = Text.assemble(
                ("* Reasoner: ", "bold red"),
                (f"{messages[idx_to_show]}\n", "white"),
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
            progress_bar.update(1)

    percent_completed = (int(num_batches_completed * batch_size) / num_samples) * 100
    percent_correct = (num_correct / int(num_batches_completed * batch_size)) * \
        100 if num_batches_completed > 0 else 0.0
    breakpoint()
    # TODO: figure out the discrepancy from percent_correct and batch_wise acuracy
    return Metrics(**{'completed': percent_completed, 'correct': percent_correct})


if __name__ == "__main__":

    from agents.generators.vllm_generator import VLLMGenerator, VLLMGeneratorConfig
    from agents.generators.argo_chat_generator import LangChainFSLGenerator, ArgoGeneratorConfig
    from agents.gsm8k.utils import batch_sample_gsm, filter_output_type, gsm_is_correct
    import pprint as pp
    from agents.utils import configure_logger
    import pprint as pp
    from functools import partial
    
    logger = configure_logger(level='info', logging_save_path='./mcts_single.log')

    config = GSMEvaluationConfig()
    dataset = read_jsonl(config.dataset_path)

    gsm_eval = partial(gsm_evaluate,
                       seed=config.seed,
                       disable_tqdm=config.disable_tqdm,
                       num_samples=config.num_samples,
                       batch_size=config.batch_size,
                       num_tries=config.num_tries)
    reasoner_registry: dict[str, BaseReasoner] = BaseReasoner.registry
    generator_cfg = VLLMGeneratorConfig()
    generator = VLLMGenerator(generator_cfg)

    name, reasoner = list(reasoner_registry.items())[2]
    reasoner = reasoner.initialize(generator, filter_output_type)

    outputs = gsm_eval(dataset=dataset, strategy=name, reasoner=reasoner)

    reasoners = [(name, reasoner.initialize(generator, filter_output_type))
                 for name, reasoner in reasoner_registry.items()]
    # TODO: FIX WM REASONER
    outputs: list[dict[str, float]] = [gsm_eval(dataset=dataset,
                                                strategy=name,
                                                reasoner=reasoner)
                                       for (name, reasoner) in reasoners]

    logger.log(pp.pformat(outputs))
