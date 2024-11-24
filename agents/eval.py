""" For running evaluation given a strategy """

from __future__ import annotations
import pprint as pp

from agents.gsm8k.utils import read_jsonl, filter_output_type
from agents.reasoners.base_reasoner import BaseReasoner
from agents.generators.vllm_generator import VLLMGeneratorConfig, VLLMGenerator
from agents.gsm8k.evaluate import gsm_evaluate, Metrics, GSMEvaluationConfig
    
    
def main(): 
    from functools import partial

    config = GSMEvaluationConfig()
    dataset = read_jsonl(config.dataset_path)
    
    gsm_eval = partial(gsm_evaluate, 
                       seed=config.seed, 
                       disable_tqdm=config.disable_tqdm, 
                       num_samples=config.num_samples, 
                       num_tries=config.num_tries)
    reasoner_registry: dict[str, BaseReasoner] = BaseReasoner.registry
    generator_cfg = VLLMGeneratorConfig()
    generator = VLLMGenerator(generator_cfg)
    reasoners = [(name, reasoner.initialize(generator, filter_output_type))
                 for name, reasoner in reasoner_registry.items()]
    # TODO: FIX WM REASONER
    outputs: list[Metrics] = [gsm_eval(dataset=dataset, 
                                                strategy=name, 
                                                reasoner=reasoner)
                                       for (name, reasoner) in reasoners]
    
    pp.pprint(outputs)
    
if __name__=="__main__": 
    
    main()