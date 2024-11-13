from __future__ import annotations

""" Tests with MCTS Node """
import pickle
from os import PathLike
import pickle
import math
from copy import deepcopy
import copy
from typing import Generic, Optional, NamedTuple, Callable, Hashable, Any, Literal, Tuple
import itertools
from abc import ABC
from collections import defaultdict
from rich.console import Console
from rich.table import Table
from io import StringIO
import numpy as np
from tqdm import trange
import random
from textwrap import dedent
from rich.tree import Tree

from agents.algorithms.tree_search.base import (SearchAlgorithm, WorldModel, SearchConfig,
                                                State, Action, Example, Trace)
from agents.utils import calculate_returns
# from agents.algorithms.tree_search.mcts_simple import MCTS, MCTSNode
from agents.gsm8k.utils import filter_output_type, gsm_is_correct
from agents.algorithms.tree_search.mcts_node import MCTSNode
from agents.reasoners.wm_reasoner import WorldModel, Actor

from agents.prompts.base_prompt_template import BasePromptTemplate
from agents.prompts.llama_prompt import GSMLlamaPromptTemplate


class MCTS(SearchAlgorithm, Generic[State, Action, Example]):
    def __init__(self,
                 output_trace_in_each_iter: bool = False,
                 w_exp: float = 1.,
                 depth_limit: int = 5,
                 n_iters: int = 10,
                 cum_reward: Callable[[list[float]], float] = sum,
                 calc_q: Callable[[list[float]], float] = np.mean,
                 simulate_strategy: str | Callable[[list[float]], int] = 'max',
                 output_strategy: str = 'max_reward',
                 disable_tqdm: bool = True,
                 node_visualizer: Callable[[MCTSNode],
                                           dict] = lambda x: x.__dict__
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
        default_simulate_strategies: dict[str, Callable[[list[float]], int]] = {
            'max': lambda x: np.argmax(x),
            'sample': lambda x: np.random.choice(len(x), p=x),
            'random': lambda x: np.random.choice(len(x)),
        }
        self.simulate_choice: Callable[[list[float]], int] = default_simulate_strategies.get(simulate_strategy,
                                                                                             simulate_strategy)
        # self.world_model = None
        # self.search_config = None
        self.output_trace_in_each_iter = output_trace_in_each_iter
        self.w_exp = w_exp
        self.depth_limit = depth_limit
        self.n_iters = n_iters
        self.cum_reward = cum_reward
        self.calc_q = calc_q
        assert output_strategy in ['max_reward', 'follow_max',
                                   'max_visit', 'max_iter',
                                   'last_iter', 'last_terminal_iter']
        self.output_strategy = output_strategy
        self._output_iter: list[MCTSNode] = None
        self._output_cum_reward = -math.inf
        self.trace_in_each_iter: list[list[MCTSNode]] = None
        self.root: Optional[MCTSNode] = None
        self.disable_tqdm = disable_tqdm
        self.node_visualizer = node_visualizer

    def _is_terminal_with_depth_limit(self, node: MCTSNode) -> bool:
        """ True if node is terminal or depth limit exceeded """
        return bool(node.is_terminal or node.depth >= self.depth_limit)

    def _uct(self, node: MCTSNode) -> float:
        """ Gets the current UCT value for the node """
        N = len(node.parent.cum_rewards)  # num times parent node visited
        n_i = max(1, len(node.cum_rewards))  # num times child node visited
        term = self.w_exp * np.sqrt(np.log(N) / n_i)

        return node.Q + term

    def _uct_select(self, node: MCTSNode) -> MCTSNode:
        """ 
        Supposing the node is fully expanded (aka max children), selects and returns the best child node (maxes UCT) out of the children 

        :node: the current node you are at in your tree search

        Note - This is called recursively in "_select" as you traverse the tree
        Note - no fast reward, so node must be fully expanded
        """
        return max(node.children, key=self._uct)

    def select(self, node: MCTSNode) -> list[MCTSNode]:
        ''' Goes through start node, and traverses via selecting best uct of children. If no children, return path as is '''
        path = []
        while True:
            path.append(node)
            if self._is_terminal_with_depth_limit(node) or len(node.children) == 0:
                return path  # NOTE: return if ...
            best_child = self._uct_select(node)
            node = best_child  # set node as best child

    def expand(self,
               node: MCTSNode,
               actor: Actor,
               world_model: WorldModel,
               question_prompt: GSMLlamaPromptTemplate,
               answer_prompt: GSMLlamaPromptTemplate,
               num_children: int
               ) -> None:
        """ Expands last node of path into d children and updates the nodes internal children attribute """
        question_prompt = copy.deepcopy(question_prompt)
        answer_prompt = copy.deepcopy(answer_prompt)
        # current prompt that encodes the history of qa interactions
        state: GSMLlamaPromptTemplate = copy.deepcopy(node.state)
        # sub_questions = [actor.act(state) for _ in range(num_children)]
        sub_questions_pkg: list[dict[str, str | list[str] | list[float]]] = [actor.act_logprobs(state) 
                                                                         for _ in range(num_children)]
        sub_questions = [sub_question['text'] for sub_question in sub_questions_pkg]
        
        # # NOTE: THIS MUST BE LOG PROBS FOR ANSWER
        # log_probs = [sub_question['log_probs'] for sub_question in sub_questions_pkg]

        children = []
        for sub_question in sub_questions:
            answer_prompt.add(**{'role': 'user', 'content': sub_question})
            sub_answer, log_prob = world_model.step_logprobs(answer_prompt).values()
            question_prompt.add(
                **{'role': 'assistant', 'content': sub_question})
            question_prompt.add(**{'role': 'user', 'content': sub_answer})
            child_node = MCTSNode(state=copy.deepcopy(question_prompt),
                                  action=sub_question,
                                  reward=np.mean(log_prob)
                                  )
            answer_prompt.pop()
            question_prompt.pop([-1, -2])
            children.append(child_node)

        node.children = children
        
    def simulate_node(self, 
                      path: list[MCTSNode], 
                      actor: Actor, 
                      world_model: WorldModel, 
                      question_prompt: GSMLlamaPromptTemplate | BasePromptTemplate, 
                      answer_prompt: GSMLlamaPromptTemplate | BasePromptTemplate,
                      max_tries: int
                      ) -> None: 
        """ Simulates a single node until end of problem """
        def episode_stop_condition(step: int, terminated: bool) -> bool: 
            """ True if max_tries exceeded or terminal"""
            return bool(step > max_tries or terminated)
        
        question_prompt = copy.deepcopy(question_prompt)
        answer_prompt = copy.deepcopy(answer_prompt)
        
        child_idx: int = random.sample(range(len(path[-1].children)), 1)[0]
        node_to_sim: MCTSNode = path[-1].children[child_idx] # for now, just first node added
        state: GSMLlamaPromptTemplate = copy.deepcopy(node_to_sim.state)
        
        step = 0
        terminated = False
        rewards = []
        while not episode_stop_condition(step, terminated):
             
            sub_question = actor.act_logprobs(state) 
            question_prompt.add(**{'role': 'assistant', 'content': sub_question})
            answer_prompt.add(**{'role': 'user', 'content': sub_question})

            sub_answer, log_prob = world_model.step_logprobs(answer_prompt).values()
            question_prompt.add('user', sub_answer)
            answer_prompt.add('assistant', sub_answer)
            
            # NOTE: finish out evaluation for big reward
            rewards.append(np.mean(log_prob))
            if filter_output_type(sub_answer)=='final_answer': 
                out = gsm_is_correct(0, sub_answer, sample)
                terminated = True
                
            
            

    # def _simulate(self, path: list[MCTSNode]):
    #     """ Goes through all of the children """
    #     node = path[-1]
    #     while True: # keep going 
    #         if self._is_terminal_with_depth_limit(node) or len(node.children) == 0: # if node is terminal or at depth limit or no children, 
    #             return
    #         child_rewards = [child.reward for child in node.children]
    #         node = node.children[self.simulate_choice(child_rewards)] # select child with best reward 
            
    #         path.append(node) # add to path then iterate again 


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

    mcts = MCTS()

    question = samples[0]['question']
    question_prompt.add('user', content=question)
    answer_prompt.add('user', content=question)

    root = MCTSNode(state=question_prompt,  # state is original question
                    action=None,
                    reward=None,
                    parent=None,
                    is_terminal=False
                    )

    generator_cfg = VLLMGeneratorConfig(temperature=0.9)
    generator = VLLMGenerator(generator_cfg)
    actor = Actor(generator)
    world_model = WorldModel(generator)

    mcts.expand(root, actor, world_model, question_prompt, answer_prompt, 5)

    breakpoint()


