""" Main Script for downloading the datasets and running evaluations """
from __future__ import annotations

from agents.utils import BaseConfig

class DatasetConfig(BaseConfig):
    dataset_name_or_path: str = ''
    

class GSMConfig(BaseConfig): 
    

# '''

# '''


# def construct_action_agent(agent_name: str,
#                            action_agent_registry: CoupledRegistry,
#                            generator: BaseLLMGenerator,
#                            parser: LLMOutputParser,
#                            solver: Optional[Callable] = None,
#                            ) -> Tuple[BaseInputPayload, BaseActionAgent]:
#     ''' Function that initializes action agent from registry and other components '''
#     action_agent_cls, cls_container, cls_payload = action_agent_registry.get(
#         agent_name).values()

#     cls_payload['generator'] = generator
#     cls_payload['llm_output_parser'] = parser
#     cls_payload['solver'] = solver

#     cls_payload = cls_container(**cls_payload)
#     agent = action_agent_cls(**asdict(cls_payload))
#     input_payload_cls = asdict(cls_payload).get('input_payload_cls')

#     return input_payload_cls(), agent


# def parse_args() -> Any:
#     parser = ArgumentParser()
#     parser.add_argument('--agent_name', type=str,
#                         default='Python', help='What kind of agent to use')
#     parser.add_argument('--generator', type=str, default='Argo',
#                         help='What kind of generator to use')
#     parser.add_argument('--parse_type', type=str, default='dict',
#                         help='What kind of parsing strategy to use for parser')
#     parser.add_argument('--use_solver', action='store_true',
#                         help='Whether to use parser or not')

#     args = parser.parse_args()
#     return args


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
    input_payload, agent = construct_action_agent('Python',
                                                  action_agent_registry,
                                                  generator,
                                                  parser
                                                  )

    input_payload

    breakpoint()


if __name__ == "__main__":

    main()
