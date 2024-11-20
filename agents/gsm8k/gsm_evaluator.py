""" Evaluator class for evaluating how well the llm agents perform """

from __future__ import annotations
from typing import Callable, Tuple, Any
import random
from tqdm import tqdm 

from agents.gsm8k.utils import read_jsonl
from agents.generators.vllm_generator import VLLMGenerator, VLLMGeneratorConfig
from agents.prompts.llama_prompt import GSMLlamaPromptTemplate
from agents.reasoners.base_reasoner import BaseReasoner
from agents.gsm8k.utils import filter_output_type, gsm_is_correct


class GSMEvaluator: 

    def __init__(self, dataset: list[dict[str, str]], reasoner: BaseReasoner, seed: int=10) -> None:
        self.dataset = dataset
        self.reasoner = reasoner
        self.seed = seed
    
    def evaluate(self, num_samples: int = 100, num_tries: int = 10) -> dict[str, Any]:
        """ Performs evaluation on N samples from GSM8K within M tries, and calculates the metrics for them """
        random.seed(self.seed)
        sample_idx = random.sample(range(len(self.dataset)), num_samples)
        samples = [self.dataset[i] for i in sample_idx]
        
        num_correct = 0
        num_completed = 0
        for idx, sample in tqdm(enumerate(samples)): 
            finished, correct = self.reasoner.generate_answer(idx, sample, num_tries)
            num_correct += finished and correct
            num_completed += finished
            self.reasoner.reset_pass()

            print(f'Correct: {num_correct}')
        percent_completed = (num_completed / num_samples) * 100
        percent_correct = (num_correct / num_completed) * 100
        
        return {'completed': percent_completed, 'correct': percent_correct}
        
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
    
    def log_prob_reward(log_probs_seq: list[float]) -> float: 
        """ Returns the average log probability"""
        return float(sum(log_probs_seq) / len(log_probs_seq))
    
    q_prompt: GSMLlamaPromptTemplate = GSMLlamaPromptTemplate('question', 1, 'question')
    a_prompt: GSMLlamaPromptTemplate = GSMLlamaPromptTemplate('answer', 1, 'answer')
    generator_cfg = VLLMGeneratorConfig()
    generator = VLLMGenerator(generator_cfg)
    # cfg = ArgoGeneratorConfig()
    # generator = LangChainFSLGenerator(cfg)
    # reasoner = WorldReasoner(generator, a_prompt, q_prompt, filter_output_type)
    reasoner = MCTSWorldReasoner(generator, a_prompt, q_prompt, filter_output_type)
    # dataset = read_jsonl('/lus/eagle/projects/FoundEpidem/bhsu/2024_research/agents/agents/data/gsm.jsonl')
    # dataset = read_jsonl('/homes/bhsu/2024_research/agents/agents/data/gsm.jsonl')
    dataset = read_jsonl('/lus/eagle/projects/FoundEpidem/bhsu/2024_research/agents/agents/data/gsm.jsonl')
    evaluator = GSMEvaluator(dataset, reasoner)
    
    metrics = evaluator.evaluate(15, 10)
    pp.pprint(metrics)
    breakpoint()
    
    prompt_2 = StandardGSMPromptTemplate()
    reasoner_2 = LLMReasoner(generator, prompt_2, filter_output_type)
    # evaluator_2 = GSMEvaluator(dataset, reasoner_2)
    # metrics_2 = evaluator_2.evaluate(15, 10)
    # pp.pprint(metrics_2)
    breakpoint()