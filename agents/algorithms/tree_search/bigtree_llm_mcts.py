""" Bigtree version of with LLM MCTS """

from __future__ import annotations
from os import PathLike
from copy import deepcopy
import copy
from typing import Generic, Optional, NamedTuple, Callable, Hashable, Any, Literal, Tuple, Union
from rich.console import Console
from rich.table import Table
from io import StringIO
import numpy as np
from tqdm import trange
import random
from textwrap import dedent
from rich.tree import Tree
from tqdm import tqdm
import math
import logging
from agents.algorithms.tree_search.base import (SearchAlgorithm, WorldModel, SearchConfig,
                                                State, Action, Example, Trace)
from agents.utils import calculate_returns
# from agents.algorithms.tree_search.mcts_simple import MCTS, MCTSNode
from agents.gsm8k.utils import filter_output_type, gsm_is_correct
from agents.algorithms.tree_search.bigtree_mcts_node import BTMCTSNode
from agents.reasoners.wm_reasoner import WorldModel, Actor

from agents.prompts.base_prompt_template import BasePromptTemplate
from agents.prompts.llama_prompt import GSMLlamaPromptTemplate


class MCTSResult(NamedTuple):
    """ Simple Container Class for MCTS Output """
    terminal_state: State
    cum_reward: float
    trace: Trace
    trace_of_nodes: list[BTMCTSNode]
    tree_state: BTMCTSNode
    trace_in_each_iter: list[list[BTMCTSNode]] = None
    tree_state_after_each_iter: list[BTMCTSNode] = None
    aggregated_result: Optional[Hashable] = None


def win_lose(win: bool,
             win_reward: float = 100,
             lose_reward: float = -50
             ) -> float:
    return win_reward if win else lose_reward


class MCTS(SearchAlgorithm, Generic[State, Action, Example]):
    def __init__(self,
                 question_prompt: BasePromptTemplate | GSMLlamaPromptTemplate,
                 answer_prompt: BasePromptTemplate | GSMLlamaPromptTemplate,
                 output_trace_in_each_iter: bool = False,
                 w_exp: float = 1.,
                 depth_limit: int = 5,
                 num_iters: int = 10,
                 cum_reward_func: Callable[[list[float]], float] = sum,
                 calc_q_func: Callable[[list[float]], float] = np.mean,
                 simulate_strategy: str | Callable[[list[float]], int] = 'max',
                 output_strategy: str = 'max_reward',
                 use_tqdm: bool = True,
                 node_visualizer: Callable[[BTMCTSNode],
                                           dict] = lambda x: x.__dict__,
                 reward_strategy: Literal['base'] = 'base',
                 logger: Optional[logging.Logger] = None
                 ) -> None:
        """
        MCTS algorithm

        :param output_trace_in_each_iter: whether to output the trace of the chosen trajectory in each iteration ; the trace is *deepcopy*-ed
                                          will also output *tree_state_after_each_iter*, which is the *deepcopy*-ed root
        :param w_exp: the weight of exploration in UCT
        :param cum_reward: the way to calculate the cumulative reward from each step. Defaults: sum
        :param calc_q: the way to calculate the Q value from histories. Defaults: np.mean
        :param simulate_strategy: simulate strategy. Options: 'max', 'sample', 'random', or use a custom function
        :param output_strategy: the way to output the result. The nodes are not *deepcopy*-ed, so the information is after all iterations
                                Options: 'max_reward': dfs on the final tree to find a trajectory with max reward using :param cum_reward:
                                         'follow_max': starting from root, choose the maximum reward child at each step. May output a non-terminal node if dead end
                                         'max_visit': the terminal node with maximum number of visits
                                         'max_iter': the trajectory with a terminal node and max reward among those in each iteration
                                         'last_iter': the last trajectory. May output a non-terminal node if the last iteration leads to a dead end
                                         'last_terminal_iter': the last trajectory with a terminal node
                                Outputs *None* if no trajectory with terminal node but required

        Note - Since no fast_reward instead of reward for unvisited children in UCT we HAVE to visit the *unvisited* children with maximum fast_reward first
        """
        super().__init__()
        rollout_strategies: dict[str, Callable[[list[float]], int]] = {
            'max': lambda x: np.argmax(x),
            'sample': lambda x: np.random.choice(len(x), p=x),
            'random': lambda x: np.random.choice(len(x)),
        }

        reward_strategies = {
            'base': (win_lose, np.mean)  # stored as
        }

        self.simulate_choice: Callable[[list[float]], int] = rollout_strategies.get(simulate_strategy,
                                                                                    simulate_strategy)

        self.output_trace_in_each_iter: bool = output_trace_in_each_iter
        self.w_exp: float = w_exp
        self.depth_limit: int = depth_limit
        self.num_iters: int = num_iters
        self.cum_reward_func: Callable = cum_reward_func
        self.calc_q_func: Callable = calc_q_func
        assert output_strategy in ['max_reward', 'follow_max',
                                   'max_visit', 'max_iter',
                                   'last_iter', 'last_terminal_iter']
        self.output_strategy = output_strategy
        self._output_iter: list[BTMCTSNode] = None
        self._output_cum_reward = -math.inf
        self.trace_in_each_iter: list[list[BTMCTSNode]] = None
        self.use_tqdm: bool = use_tqdm
        self.node_visualizer = node_visualizer
        self.terminal_reward_strategy, self.reward_strategy = reward_strategies.get(
            reward_strategy)

        # base question prompt used for expansion and simulation
        self._question_prompt_base = question_prompt
        # base answer prompt used for expansion and simulation
        self._answer_prompt_base = answer_prompt

        self.logger = logger

    def _is_terminal_with_depth_limit(self, node: BTMCTSNode) -> bool:
        """ True if node is terminal or depth limit exceeded """
        return bool(node.is_terminal
                    or node.depth >= self.depth_limit)

    def _uct(self, node: BTMCTSNode) -> float:
        """ 
        Gets the current UCT value for the node 

        :param node: 

         - Note: cum_rewards = num full rounds (expansion -> simulation -> backprop) involving that node

         - N = Calculates number times parent node visited via cum_rewards

         - n_i = number times child node visited via cum_rewards
        """
        N = len(node.parent.cum_rewards)  # num times parent node visited -
        n_i = max(1, len(node.cum_rewards))  # num times child node visited
        term = self.w_exp * np.sqrt(np.log(N) / n_i)  # left term in UCT

        return node.Q + term

    def _uct_select(self, node: BTMCTSNode) -> BTMCTSNode:
        """ 
        Supposing the node is fully expanded (aka max children), selects and returns the best child node (maxes UCT) out of the children 

        :node: the current node you are at in your tree search

        Note - This is called recursively in "_select" as you traverse the tree
        Note - no fast reward, so node must be fully expanded
        """
        return max(node.children, key=self._uct)

    def select(self, node: BTMCTSNode) -> list[BTMCTSNode]:
        ''' Goes through start node, and traverses via selecting best uct of children. If no children or terminal or depth-limit hit, return path as is '''
        path = []
        while True:
            path.append(node)
            if self._is_terminal_with_depth_limit(node) or len(node.children) == 0:
                return path
            best_child = self._uct_select(node)
            node = best_child  # set node as best child

    # NOTE: since q/a prompt is used just for copying state of node, maybe just use it as an attr container
    def expand(self,
               node: BTMCTSNode,
               actor: Actor,
               world_model: WorldModel,
               num_children: int,
               sample: dict,
               sample_idx: int
               ) -> None:
        """ Expands last node of path into d children and updates the nodes internal children attribute """
        # copy prompt history from node state
        self._question_prompt_base.copy_history(node.state)
        self._answer_prompt_base.copy_history(node.state)

        # get actions
        sub_questions = [actor.act(node.state) for _ in range(num_children)]

        children = []
        for sub_question in sub_questions:

            self._answer_prompt_base.add(
                **{'role': 'user', 'content': sub_question})
            sub_answer, log_prob = world_model.step_logprobs(
                self._answer_prompt_base).values()
            self._question_prompt_base.add(
                **{'role': 'assistant', 'content': sub_question})
            self._question_prompt_base.add(
                **{'role': 'user', 'content': sub_answer})

            if filter_output_type(sub_answer) == 'final_answer':
                out, message = gsm_is_correct(sample_idx, sub_answer, sample)
                if self.logger:
                    self.logger.info(message)
                else:
                    print(message)
                reward: float | int = self.terminal_reward_strategy(out)
                terminated = True
            else:
                reward = self.reward_strategy(
                    log_prob)  # get log_prob as reward
                terminated = False  # set as not done

            child_node = BTMCTSNode(state=copy.deepcopy(self._question_prompt_base),
                                    action=sub_question,
                                    reward=reward,
                                    parent=node,
                                    is_terminal=terminated,
                                    calc_q=self.calc_q_func
                                    )
            self._answer_prompt_base.pop()
            self._question_prompt_base.pop([-1, -2])
            children.append(child_node)

        node.children = children
        self._question_prompt_base.reset()
        self._answer_prompt_base.reset()

    def simulate_node(self,
                      path: list[BTMCTSNode],
                      actor: Actor,
                      world_model: WorldModel,
                      max_tries: int,
                      sample: dict,
                      sample_idx: int
                      ) -> bool:
        """ Simulates a single node until end of problem and returns a flag if successfully simulated or not """
        # randomly simulate child
        child_idx: int = random.sample(
            range(len(path[-1].children)), 1)[0]  # randomly choose a child
        node_to_sim: BTMCTSNode = path[-1].children[child_idx]

        self._question_prompt_base.copy_history(node_to_sim.state)
        self._answer_prompt_base.copy_history(node_to_sim.state)

        step, terminated, exceeds_limit = 0, False, False
        rollout_rewards = []
        while not terminated:
            # selecting action aka sub-question
            sub_question = actor.act(self._question_prompt_base)
            self._question_prompt_base.add(
                **{'role': 'assistant', 'content': sub_question})
            self._answer_prompt_base.add(
                **{'role': 'user', 'content': sub_question})
            # world model returns next state aka sub-answer
            sub_answer, log_prob = world_model.step_logprobs(
                self._answer_prompt_base).values()
            self._question_prompt_base.add(
                **{'role': 'user', 'content': sub_answer})
            self._answer_prompt_base.add(
                **{'role': 'assistant', 'content': sub_answer})

            rollout_rewards.append(self.reward_strategy(log_prob))
            if filter_output_type(sub_answer) == 'final_answer':
                out, message = gsm_is_correct(sample_idx, sub_answer, sample)
                rollout_rewards.append(self.terminal_reward_strategy(out))
                terminated = True
                if self.logger:
                    self.logger.info(message)
                else:
                    print(message)
                    
            # def exceeds_prompt_limit(prompts: tuple[Union[BasePromptTemplate, GSMLlamaPromptTemplate]]) -> bool: 
            #     return any(agent.prompt_exceeds_limit(prompt) for agent, prompt in zip((actor, world_model), ))

            # def exceeds_prompt_limit(agents: tuple, prompts: tuple) -> bool:
            #     """Check if any agent exceeds the prompt limit for the corresponding prompt."""
            #     return any(agent.prompt_exceeds_limit(prompt) for agent, prompt in zip(agents, prompts))
            # test for if any of the prompts exceed input token limit before next round
            exceeds_limit: bool = any([agent.prompt_exceeds_limit(prompt)
                                       for agent, prompt in zip(
                (actor, world_model),
                (self._question_prompt_base,
                 self._answer_prompt_base)
            )
            ])
            step += 1
            if step > max_tries or exceeds_limit:
                return False  # flag for skipping backprop; assumption - parsing error
        # reset prompts to base again
        self._question_prompt_base.reset()
        self._answer_prompt_base.reset()
        # idea return flag, if flag then skip backpropagation and then move to next
        rollout_reward = sum(rollout_rewards)
        node = path[-1].children[child_idx]
        node.reward = rollout_reward
        path.append(node)

        return True

    def back_propagate(self, path: list[BTMCTSNode]) -> float:
        """ 
        Updates each node in the path with the cumulative rewards from rollout and returns the updated path and the cum_reward for the root 

        ex. leaf node gets rollout reward 
        leaf node - 1 gets rollout reward + own reward 
        leaf node - 2 gets rollout reward + leaf node -1 + own reward 
        ...

        :param path - list[MCTSNode]: list of nodes corresponding to search path 
        :param child_idx - int: Inside the leaf node, the idx of its expanded child node we simulated
        """
        rewards = []  # holds rewards for each node
        for node in reversed(path):  # leaf --> root
            # ex. leaf: rewards = [100]; leaf-1: rewards = [100, 10]; leaf-2: rewards = [100, 10, 15], ...
            rewards.append(node.reward)
            # NOTE: work-around for node.reward = None => we filter this out
            rewards = list(filter(lambda x: x != None, rewards))
            # self.cum_rewards callable sum; ex. sum([10, 100]), sum([15, 10, 100])
            cum_reward = self.cum_reward_func(rewards[::-1])
            # node.cum_rewards stores summed rewards for one iteration; ex. (leaf-1).cum_reward = 110
            node.cum_rewards.append(cum_reward)

        return cum_reward

    def iterate(self,
                node: BTMCTSNode,
                actor: Actor,
                world_model: WorldModel,
                num_children: int,
                sample: dict,
                sample_idx: int,
                max_tries: int
                ) -> Union[None, list[BTMCTSNode]]:
        """ Runs one single iteration of MCTS on input node using actor - world model strategy """
        path = self.select(node)
        # cum_reward = path[-1].cum_rewards
        # if leaf is not terminal or exceeds depth
        if not self._is_terminal_with_depth_limit(path[-1]):
            self.expand(path[-1], actor, world_model,
                        num_children, sample, sample_idx)
            simulated = self.simulate_node(
                path, actor, world_model, max_tries, sample, sample_idx)  # simulate the path
        else:
            simulated = False  # if terminal or depth limit, dont simulate

        if simulated:
            # if everything fine, backprop
            cum_reward = self.back_propagate(path)
            # set attr to cumulative reward
            self._output_cum_reward = cum_reward
            # if last node is terminal and finds the path with the best cum_reward
            if self.output_strategy == 'max_iter' \
                    and path[-1].is_terminal \
                    and cum_reward > self._output_cum_reward:
                self._output_iter = path
            # only returns the last iteration
            elif self.output_strategy == 'last_iter':
                self._output_iter = path
            # only returns the last iteration where the leaf node is terminal
            elif self.output_strategy == 'last_terminal_iter' \
                    and path[-1].is_terminal:
                self._output_iter = path

            return path

        return None


if __name__ == "__main__":
    from agents.reasoners.wm_reasoner import Actor, WorldModel
    from agents.gsm8k.utils import read_jsonl, batch_sample_gsm
    from agents.prompts.llama_prompt import GSMLlamaPromptTemplate
    from agents.generators.vllm_generator import VLLMGenerator, VLLMGeneratorConfig

    data_path = '/lus/eagle/projects/FoundEpidem/bhsu/2024_research/agents/agents/data/gsm.jsonl'
    batch_size = 16

    dataset = read_jsonl(data_path)
    samples = batch_sample_gsm(dataset, batch_size)

    question_prompt: GSMLlamaPromptTemplate = GSMLlamaPromptTemplate(
        'question', 1, 'question')
    answer_prompt: GSMLlamaPromptTemplate = GSMLlamaPromptTemplate(
        'answer', 1, 'answer')

    question = samples[0]['question']
    question_prompt.add('user', content=question)
    answer_prompt.add('user', content=question)

    root = BTMCTSNode(state=question_prompt,  # state is original question
                      action=None,
                      reward=None,
                      parent=None,
                      is_terminal=False
                      )

    mcts = MCTS(question_prompt=question_prompt, answer_prompt=answer_prompt)

    generator_cfg = VLLMGeneratorConfig(temperature=0.9)
    generator = VLLMGenerator(generator_cfg)
    actor = Actor(generator)
    world_model = WorldModel(generator)

    mcts.iterate(root, actor, world_model, 3, samples[0], 0, 10)

    breakpoint()
