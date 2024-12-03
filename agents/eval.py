""" For running parsl evaluation given a strategy """

from __future__ import annotations
from typing import Any, TypeVar, Optional
from parsl import python_app
from pydantic import Field
from argparse import ArgumentParser
import logging

from agents.generators import VLLMGeneratorConfig
from agents.gsm8k import GSM8KProblem, GSMEvaluationConfig, Metrics
from agents.agent_parsl import PolarisConfig
from agents.utils import BaseConfig

T = TypeVar('T')

class MasterConfig(BaseConfig):
    """ Master Config """
    dataset_path: str = Field(
        '/lus/eagle/projects/FoundEpidem/bhsu/2024_research/agents/agents/data/gsm.jsonl')
    num_chunks: int = Field(
        4
    )
    eval_config: GSMEvaluationConfig = Field(
        default_factory=GSMEvaluationConfig
    )
    parsl_config: PolarisConfig = Field(
        default_factory=PolarisConfig
    )
    
def truncate_dataset(dataset: list[GSM8KProblem], 
                     batch_size: int, 
                     logger: Optional[logging.Logger]=None) -> list[GSM8KProblem]:
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


def split_dataset(dataset: list[GSM8KProblem], num_chunks: int, batch_size: int) -> list[list[GSM8KProblem]]:
    """Chunks dataset into equally sized chunks divisible by batch size."""
    # Calculate the total number of samples to use (divisible by batch_size * num_chunks)
    total_samples = (len(dataset) // (batch_size * num_chunks)) * (batch_size * num_chunks)
    # Adjust the dataset to use only the total_samples
    dataset = dataset[:total_samples]
    # Calculate the chunk size (equal for all chunks and divisible by batch_size)
    chunk_size = total_samples // num_chunks

    return [dataset[i * chunk_size: (i + 1) * chunk_size] 
            for i in range(num_chunks)]


@python_app
def parsl_batch_generate_answer(dataset_chunk: list[GSM8KProblem], 
                                eval_config: GSMEvaluationConfig, 
                                generator_config: VLLMGeneratorConfig
                                ) -> Metrics:
    """ Runs batch generation on parsl instance """
    from agents.reasoners import BaseReasoner, LLMReasoner, WorldReasoner, MCTSWorldReasoner
    from agents.generators import VLLMGenerator, VLLMGeneratorConfig
    from agents.gsm8k import GSM8KProblem, Metrics
    from agents.gsm8k import filter_output_type, batch_gsm_evaluate, read_jsonl_dataset
    from dataclasses import asdict

    # Reconstruct the generator and reasoner inside the function
    generator = VLLMGenerator(generator_config)
    reasoner_registry = BaseReasoner.registry
    reasoner_cls = reasoner_registry[eval_config.strategy]
    reasoner = reasoner_cls.initialize(generator, filter_output_type)
    num_samples = len(dataset_chunk)

    # Call batch_gsm_evaluate
    metrics = batch_gsm_evaluate(
        strategy=eval_config.strategy,
        dataset=dataset_chunk,
        reasoner=reasoner,
        seed=eval_config.seed,
        disable_tqdm=eval_config.disable_tqdm,
        num_samples=num_samples,
        batch_size=eval_config.batch_size,
        num_tries=eval_config.num_tries,
        logger=None
    )
    return metrics

def standard_batch_generate_answer(dataset: list[GSM8KProblem], 
                                   eval_config: GSMEvaluationConfig, 
                                   generator_config: VLLMGeneratorConfig
                                   ) -> Metrics: 
    """ Runs standard batch generation """
    from agents.reasoners import BaseReasoner, LLMReasoner, WorldReasoner, MCTSWorldReasoner
    from agents.generators import VLLMGenerator, VLLMGeneratorConfig
    from agents.gsm8k import GSM8KProblem, Metrics
    from agents.gsm8k import filter_output_type, batch_gsm_evaluate, read_jsonl_dataset
    from dataclasses import asdict
    
    # Reconstruct the generator and reasoner inside the function
    generator = VLLMGenerator(generator_config)
    reasoner_registry = BaseReasoner.registry
    reasoner_cls = reasoner_registry[eval_config.strategy]
    reasoner = reasoner_cls.initialize(generator, filter_output_type)
    num_samples = len(dataset)

    # Call batch_gsm_evaluate
    metrics = batch_gsm_evaluate(
        strategy=eval_config.strategy,
        dataset=dataset,
        reasoner=reasoner,
        seed=eval_config.seed,
        disable_tqdm=eval_config.disable_tqdm,
        num_samples=num_samples,
        batch_size=eval_config.batch_size,
        num_tries=eval_config.num_tries,
        logger=None
    )
    return metrics


def parse_args() -> Any:

    arg_parser = ArgumentParser()
    arg_parser.add_argument('--dataset_path',
                            type=str,
                            default='/lus/eagle/projects/FoundEpidem/bhsu/2024_research/agents/agents/data/gsm.jsonl'
                            )
    arg_parser.add_argument('--logging_save_path',
                            type=str,
                            default='./parsl_gsm_outputs.log'
                            )
    arg_parser.add_argument('--master_config_path',
                            type=str,
                            default='/lus/eagle/projects/FoundEpidem/bhsu/2024_research/agents/agents/master_config.yaml'
                            )
    arg_parser.add_argument('--use_parsl', 
                            action='store_true',
                            default=False
                            )
    arg_parser.add_argument('--model_path', 
                            type=str, 
                            default='/lus/eagle/projects/FoundEpidem/bhsu/2024_research/agents/agents/Meta-Llama-3-8B-Instruct-W8A8-Dynamic-Per-Token'
                            )
    arg_parser.add_argument('--dtype', 
                            type=str, 
                            default='bfloat16')
    arg_parser.add_argument('--strategy', 
                            type=str, 
                            default='base')
    return arg_parser.parse_args()


if __name__ == "__main__":
    import pprint as pp
    import parsl
    from parsl.concurrent import ParslPoolExecutor
    from dataclasses import asdict
    from itertools import repeat
    
    from agents.gsm8k import read_jsonl_dataset
    from agents.utils import configure_logger

    args = parse_args()
    master_config = MasterConfig.from_yaml(args.master_config_path)
    master_config.dataset_path = args.dataset_path
    
    eval_config: GSMEvaluationConfig = master_config.eval_config
    eval_config.dataset_path = args.dataset_path
    
    generator_config: VLLMGeneratorConfig = VLLMGeneratorConfig(llm_name=args.model_path,
                                                                dtype=args.dtype)
    
    logger = configure_logger('info', logging_save_path=args.logging_save_path)
    dataset = read_jsonl_dataset(master_config.dataset_path)
    dataset = truncate_dataset(dataset, eval_config.batch_size)
    
    if args.use_parsl: 
        parsl_config = master_config.parsl_config.get_config(
            '/lus/eagle/projects/FoundEpidem/bhsu/2024_research/agents/agents/parsl_outputs'
        )
        parsl.load(parsl_config) 
        dataset_chunks: list[GSM8KProblem] = split_dataset(
            dataset, master_config.num_chunks, eval_config.batch_size)

        assert all([len(dataset_chunk) % eval_config.batch_size == 0 for dataset_chunk in dataset_chunks]), f'''
        Dataset chunks are not divisible by batch size 
        '''
        # Create a list to hold futures
        futures = []
        for dataset_chunk in dataset_chunks:
            future = parsl_batch_generate_answer(dataset_chunk, eval_config, generator_config)
            futures.append(future)

        metrics_list: list[Metrics] = [future.result() for future in futures]
    
        for idx, metric in enumerate(metrics_list): 
            
            logger.info(f'Results for Process {idx + 1}:\n {pp.pformat(asdict(metric))}\n\n')
            
        total_correct = sum(m['num_correct'] for m in metrics_list)
        total_completed = sum(m['num_completed'] for m in metrics_list)
        total_num_samples = sum(m['num_total'] for m in metrics_list)
        overall_percent_completed = (total_completed / total_num_samples) * 100
        overall_percent_correct = (total_correct / total_completed) * 100 if total_completed > 0 else 0.0
        overall_metrics = Metrics(
            percent_completed=overall_percent_completed,
            percent_correct=overall_percent_correct,
            num_correct=total_correct,
            num_completed=total_completed,
            num_total=total_num_samples
        )
        
        logger.info(f'Aggregate Results:\n\n{asdict(overall_metrics)}')
        
    else: 
        metrics = standard_batch_generate_answer(dataset, eval_config, generator_config)
        logger.info(f'Results from running:\n\n{asdict(metrics)}')
