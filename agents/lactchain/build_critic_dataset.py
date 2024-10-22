''' Used to Build Up a Huggingface dataset with an actor and critic'''

from __future__ import annotations

from typing import Optional, Callable, Any, Tuple
from argparse import ArgumentParser
from dataclasses import asdict

from agents.registry import CoupledRegistry
from agents.lactchain import lactchain_agent_registry, lactchain_prompt_registry

# from agents.llm_agents import BaseActionAgent
from agents.lactchain.agents.strategist import LactChainStrategyChain
from agents.prompts.base_prompt import BaseInputPayload
from agents.parsers import LLMOutputParser, get_parser
from agents.generators import get_generator, BaseLLMGenerator


def construct_strategy_agent(agent_name: str,
                             lactchain_agent_registry: CoupledRegistry,
                             lactchain_prompt_registry: CoupledRegistry,
                             generator: BaseLLMGenerator,
                             parser: LLMOutputParser,
                             solver: Optional[Callable] = None,
                             ) -> Tuple[BaseInputPayload, LactChainStrategyChain]:
    ''' Function that initializes action agent from registry and other components '''
    action_agent_cls, cls_container, cls_payload = lactchain_agent_registry.get(
        agent_name).values()
    
    input_payload_cls, output_payload_cls = lactchain_prompt_registry.get(agent_name).values()

    # make new fields to make them
    cls_payload['generator'] = generator
    cls_payload['llm_output_parser'] = parser
    cls_payload['solver'] = solver
    cls_payload['input_payload_cls'] = input_payload_cls
    cls_payload['output_payload_cls'] = output_payload_cls

    cls_payload = cls_container(**cls_payload)
    agent = action_agent_cls(**asdict(cls_payload))
    input_payload_cls = asdict(cls_payload).get('input_payload_cls')

    return input_payload_cls(), agent


def parse_args() -> Any:
    parser = ArgumentParser()
    parser.add_argument('--agent_name', type=str,
                        default='Python', help='What kind of agent to use')
    parser.add_argument('--generator', type=str, default='vllm',
                        help='What kind of generator to use')
    parser.add_argument('--parse_type', type=str, default='dict',
                        help='What kind of parsing strategy to use for parser')
    parser.add_argument('--use_solver', action='store_true',
                        help='Whether to use parser or not')

    args = parser.parse_args()
    return args


def main():
    from agents import action_agent_registry
    args = parse_args()
    agent_names = action_agent_registry.get_supported_agents()
    assert args.agent_name in agent_names, f'''
    Agent {args.agent_name} is nor supported, please choose from available registered agents: 
    {agent_names}
    '''

    generator = get_generator(args.generator)
    parser = get_parser(args.parse_type)
    breakpoint()
    input_payload, agent = construct_action_agent('Python',
                                                  action_agent_registry,
                                                  generator,
                                                  parser
                                                  )


if __name__ == "__main__":

    main()
