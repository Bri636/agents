""" Evaluator class for evaluating how well the llm agents perform """

from __future__ import annotations
from typing import Callable, Tuple, Any
import random, time
from tqdm import tqdm
from tqdm.rich import tqdm
from dataclasses import dataclass
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
from agents.utils import BaseConfig, register_strategy
# import types 
from agents.gsm8k import T

@dataclass
class Metrics: 
    completed: float
    correct: float 

class GSMEvaluator: 
    
    reasoner_strategies = {}

    def __init__(self, 
                 strategy: str,
                 dataset: list[dict[str, str]], 
                 reasoner: BaseReasoner, 
                 seed: int=10, 
                 disable_tqdm: bool = True
                 ) -> None:
        
        self.strategy = strategy
        self.dataset = dataset
        self.reasoner = reasoner
        self.seed = seed
        self.disable_tqdm = disable_tqdm
    
    def evaluate(self, num_samples: int = 100, num_tries: int = 10) -> dict[str, Any]:
        """ Performs evaluation on N samples from GSM8K within M tries, and calculates the metrics for them """
        random.seed(self.seed)
        sample_idx = random.sample(range(len(self.dataset)), num_samples)
        samples = [self.dataset[i] for i in sample_idx]
        
        console = Console()
        console.rule(f'Running Eval on {self.strategy} Reasoner', style="bold", characters='=')
        map(lambda _: console.rule(f'', style="bold", characters='='), range(2))
        time.sleep(1)
        
        num_correct = 0
        num_completed = 0
        with tqdm(total=num_samples, disable=self.disable_tqdm, desc=f"GSM Evaluation - {num_samples} Samples", leave=False) as progress_bar:
            for idx, sample in enumerate(samples):
                finished, correct, message, panel = self.reasoner.generate_answer(idx, sample, num_tries)
                num_correct += finished and correct
                num_completed += finished
                self.reasoner.reset_pass()
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
        percent_correct = (num_correct / num_completed) * 100 if num_completed > 0 else 0.0
        
        return {'completed': percent_completed, 'correct': percent_correct}
    
    @classmethod
    def initialize(cls: T, 
                   strategy: str, 
                   generator: BaseLLMGenerator, 
                   dataset_path: str, 
                   seed: int = 10, 
                   disable_tqdm: bool = True,
                   ) -> T:
        dataset = read_jsonl(dataset_path)
        constructor_func: Callable = cls.reasoner_strategies.get(strategy)
        if not constructor_func: 
            raise ValueError(f"Eval. Strategy '{strategy}' does not exist. Choose from {list(cls.reasoner_strategies.keys())}")
        reasoner = constructor_func(generator)
        return cls(strategy, dataset, reasoner, seed, disable_tqdm)
    
    @staticmethod
    @register_strategy(reasoner_strategies, name='world_model')
    def world_model_reasoner(generator: BaseLLMGenerator) -> WorldReasoner: 
        question_prompt = GSMLlamaPromptTemplate('question', 1, 'question')
        answer_prompt = GSMLlamaPromptTemplate('answer', 1, 'answer')
        reasoner = WorldReasoner(generator, answer_prompt, question_prompt, filter_output_type)
        return reasoner
    
    @staticmethod
    @register_strategy(reasoner_strategies, name='mcts_world_model')
    def mcts_world_model_reasoner(generator: BaseLLMGenerator) -> MCTSWorldReasoner: 
        question_prompt = GSMLlamaPromptTemplate('question', 1, 'question')
        answer_prompt = GSMLlamaPromptTemplate('answer', 1, 'answer')
        reasoner = MCTSWorldReasoner(generator, answer_prompt, question_prompt, filter_output_type)
        return reasoner
    
    @staticmethod
    @register_strategy(reasoner_strategies, name='base')
    def base_reasoner(generator: BaseLLMGenerator) -> MCTSWorldReasoner: 
        prompt = StandardGSMPromptTemplate()
        reasoner = LLMReasoner(generator, prompt, filter_output_type)
        return reasoner
        
if __name__ == "__main__": 
    
    from agents.generators.vllm_generator import VLLMGenerator, VLLMGeneratorConfig
    from agents.generators.argo_chat_generator import LangChainFSLGenerator, ArgoGeneratorConfig
    from agents.gsm8k.utils import batch_sample_gsm, filter_output_type, gsm_is_correct
    import pprint as pp
    from agents.reasoners.wm_reasoner import WorldReasoner
    from agents.reasoners.reasoner import LLMReasoner
    from agents.prompts.standard_prompt import StandardGSMPromptTemplate
    from agents.reasoners.wm_mcts_reasoner import MCTSWorldReasoner
    
    random.seed(10)
    dataset_path = '/lus/eagle/projects/FoundEpidem/bhsu/2024_research/agents/agents/data/gsm.jsonl'
    generator_cfg = VLLMGeneratorConfig()
    generator = VLLMGenerator(generator_cfg)
    
    seed, disable_tqdm = 10, False
    num_samples, num_tries = 25, 10
    
    base_evaluator = GSMEvaluator.initialize('base', generator, dataset_path, 
                                             seed=seed, disable_tqdm=disable_tqdm)
    wm_evaluator = GSMEvaluator.initialize('world_model', generator, dataset_path, 
                                           seed=seed, disable_tqdm=disable_tqdm)
    mcts_wm_evaluator = GSMEvaluator.initialize('mcts_world_model', generator, dataset_path, 
                                            seed=seed, disable_tqdm=disable_tqdm)
    
    base_metrics = base_evaluator.evaluate(num_samples, num_tries)
    wm_metrics = wm_evaluator.evaluate(num_samples, num_tries)
    mcts_wm_metrics = mcts_wm_evaluator.evaluate(num_samples, num_tries)
    
    
    # evaluators: list[] = GSMEvaluator.initialize()
    
    # metrics: list[dict[str, str]] = [evaluator.evaluate(num_samples, num_tries) for evaluator in ]
    
    
    # q_prompt: GSMLlamaPromptTemplate = GSMLlamaPromptTemplate('question', 1, 'question')
    # a_prompt: GSMLlamaPromptTemplate = GSMLlamaPromptTemplate('answer', 1, 'answer')
    # generator_cfg = VLLMGeneratorConfig()
    # generator = VLLMGenerator(generator_cfg)
    # # cfg = ArgoGeneratorConfig()
    # # generator = LangChainFSLGenerator(cfg)
    # # reasoner = WorldReasoner(generator, a_prompt, q_prompt, filter_output_type)
    # reasoner = MCTSWorldReasoner(generator, a_prompt, q_prompt, filter_output_type)
    # # dataset = read_jsonl('/lus/eagle/projects/FoundEpidem/bhsu/2024_research/agents/agents/data/gsm.jsonl')
    # # dataset = read_jsonl('/homes/bhsu/2024_research/agents/agents/data/gsm.jsonl')
    # dataset = read_jsonl('/lus/eagle/projects/FoundEpidem/bhsu/2024_research/agents/agents/data/gsm.jsonl')
    # evaluator = GSMEvaluator(dataset, reasoner)
    
    # metrics = evaluator.evaluate(15, 10)
    # pp.pprint(metrics)
    
    # prompt_2 = StandardGSMPromptTemplate()
    # reasoner_2 = LLMReasoner(generator, prompt_2, filter_output_type)
    # # evaluator_2 = GSMEvaluator(dataset, reasoner_2)
    # # metrics_2 = evaluator_2.evaluate(15, 10)
    # # pp.pprint(metrics_2)
    # breakpoint()