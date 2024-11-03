''' Used to Build Up a Huggingface dataset with an actor and critic'''

from __future__ import annotations

from typing import Optional, Callable, Any, Tuple, Union
from argparse import ArgumentParser
from dataclasses import asdict
from functools import partial
from textwrap import dedent

from agents.registry import CoupledRegistry
from agents.llm_agents.strategist import LactChainStrategyChain
from agents.llm_agents.actor import LactChainActorChain

from agents.prompts import get_instruction
from agents.extra.prompts.strategy_prompt_template import StrategyInputPayload
from agents.extra.prompts.action_prompt_template import ActorInputPayload
from agents.parsers import LLMOutputParser, get_parser
from agents.generators import get_generator, BaseLLMGenerator


def construct_agent(agent_name: str,
                    agent_registry: CoupledRegistry,
                    prompt_registry: CoupledRegistry,
                    generator: BaseLLMGenerator,
                    parser: Optional[LLMOutputParser] = None,
                    ) -> Tuple[Union[StrategyInputPayload, ActorInputPayload],
                               Union[LactChainStrategyChain, LactChainActorChain]]:
    ''' Function that initializes strategy agent from registry and other components '''
    action_agent_cls, cls_container, cls_payload = agent_registry.get(
        agent_name).values()

    prompt_template_cls, payloads = prompt_registry.get(agent_name).values()
    input_payload_cls = payloads['input']

    # make new fields to make them
    cls_payload['llm_output_parser'] = parser
    cls_payload['prompt_template_cls'] = prompt_template_cls
    cls_payload['input_payload_cls'] = input_payload_cls

    cls_payload = cls_container(**cls_payload)
    agent = action_agent_cls(generator=generator, **asdict(cls_payload))
    input_payload_cls = asdict(cls_payload).get('input_payload_cls')

    return input_payload_cls, agent


def parse_args() -> Any:
    parser = ArgumentParser()
    parser.add_argument('--generator', type=str, default='vllm',
                        help='What kind of generator to use')
    parser.add_argument('--parse_type', type=str, default='dict',
                        help='What kind of parsing strategy to use for parser')
    parser.add_argument('--use_solver', action='store_true',
                        help='Whether to use parser or not')

    args = parser.parse_args()
    return args


def main():
    from agents import agent_registry, prompt_registry

    args = parse_args()
    agent_names = agent_registry.get_supported_agents()
    
    generator = get_generator(args.generator)
    parser = get_parser(args.parse_type)

    strategy_payload_cls, strategist = construct_agent("Strategist",
                                                       agent_registry,
                                                       prompt_registry,
                                                       generator,
                                                       parser
                                                       )

    strategy_inputs = [{'task': 'You are solving a math problem',
                        'context': dedent('''
                    Return a strategy or thinking pattern you would use to solve problems of this kind
                    ''')}] * 4

    strategist_payloads = [strategy_payload_cls(**test_input)
                           for test_input in strategy_inputs]

    strategist_outputs: list[str] = strategist.batch_generate(
        strategist_payloads)
    
    for x in strategist_outputs: 
        print(x)
        
    action_inputs = [{'goal': 'Your goal is to solve the math problem given to you.',
                      'task': 'What is the gcd(54, 21)?',
                      'strategy': llm_output} for llm_output in strategist_outputs]

    instruction = get_instruction('COT')

    action_payload_cls, actor = construct_agent("Actor",
                                                agent_registry,
                                                prompt_registry,
                                                generator,
                                                parser
                                                )

    actor_payloads = [action_payload_cls(**action_input)
                      for action_input in action_inputs]

    for action_payload in actor_payloads:
        setattr(action_payload, 'instruction', instruction)

    breakpoint()
    actor_outputs: list[str] = actor.batch_generate(actor_payloads)
    breakpoint()

if __name__ == "__main__":

    main()
