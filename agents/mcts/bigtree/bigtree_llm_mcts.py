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
from tqdm.rich import trange, tqdm
import random
from textwrap import dedent
from rich.tree import Tree
import math
import logging
from agents.mcts.base import (SearchAlgorithm, WorldModel, SearchConfig,
                              State, Action, Example, Trace)
from agents.utils import calculate_returns
# from agents.algorithms.tree_search.mcts_simple import MCTS, MCTSNode
from agents.gsm8k.utils import filter_output_type, gsm_is_correct
from agents.mcts.bigtree.bigtree_mcts_node import BTMCTSNode
from agents.reasoners.wm_reasoner import WorldModel, Actor

from agents.prompts.base_prompt_template import BasePromptTemplate
from agents.prompts.llama_prompt import GSMLlamaPromptTemplate
from agents.mcts.bigtree.mcts_utils import SearchStrategies
# from agents.utils import configure_logger

# importing types 
from agents.mcts.bigtree import Prompt, Computable

def win_lose(win: bool,
             win_reward: float = 100,
             lose_reward: float = -50
             ) -> float:
    return win_reward if win else lose_reward

class MCTS:
    def __init__(self,
                 question_prompt: Prompt,
                 answer_prompt: Prompt,
                 output_trace_in_each_iter: bool = False,
                 w_exp: float = 1.,
                 depth_limit: int = 5,
                 num_iters: int = 10,
                 cum_reward_func: Callable[[Computable], float] = sum,
                 calc_q_func: Callable[[Computable], float] = np.mean,
                 simulate_strategy: str | Callable[[Computable], int] = 'max',
                 output_strategy: str = 'max_reward',
                 use_tqdm: bool = True,
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
        self.trace_in_each_iter = []
        self.use_tqdm: bool = use_tqdm
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

    def expand(self,
               node: BTMCTSNode,
               actor: Actor,
               world_model: WorldModel,
               num_children: int,
               sample: dict,
               sample_idx: int, 
               verbose: bool = False
               ) -> None:
        """ Expands last node of path into d children and updates the nodes internal children attribute """
        # copy prompt history from node state
        self._question_prompt_base.copy_history(node.state)
        self._answer_prompt_base.copy_history(node.state)

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
                if verbose:
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
                      sample_idx: int, 
                      verbose: bool = False
                      ) -> bool:
        """ Simulates a single node until end of problem and returns a flag if successfully simulated or not """
        # randomly select child
        child_idx: int = random.sample(range(len(path[-1].children)), 1)[0]  # randomly choose a child
        node_to_sim: BTMCTSNode = path[-1].children[child_idx]
        # copy child state for simulation
        self._question_prompt_base.copy_history(node_to_sim.state)
        self._answer_prompt_base.copy_history(node_to_sim.state)

        rollout_rewards = []
        for _ in range(1, max_tries + 1):
            # generating sub-question
            sub_question = actor.act(self._question_prompt_base)
            self._question_prompt_base.add(role='assistant', content=sub_question)
            self._answer_prompt_base.add(role='user', content=sub_question)
            # generating sub-answer
            step_result = world_model.step_logprobs(self._answer_prompt_base)
            sub_answer = step_result.get('text')
            log_prob = step_result.get('log_probs')
            # updating prompts
            self._question_prompt_base.add(role='user', content=sub_answer)
            self._answer_prompt_base.add(role='assistant', content=sub_answer)
            # collecting reward
            rollout_rewards.append(self.reward_strategy(log_prob))
            # Check for termination condition
            if filter_output_type(sub_answer) == 'final_answer':
                out, message = gsm_is_correct(sample_idx, sub_answer, sample)
                rollout_rewards.append(self.terminal_reward_strategy(out))
                if verbose:
                    print(message)
                # reset prompts to base again
                self._question_prompt_base.reset()
                self._answer_prompt_base.reset()
                # idea return flag, if flag then skip backpropagation and then move to next
                rollout_reward = sum(rollout_rewards)
                node = path[-1].children[child_idx]
                node.reward = rollout_reward
                path.append(node)
                # successful simulation
                return True
            # exit if prompt exceeds limit aka its a run-on
            agents = (actor, world_model)
            prompts = (self._question_prompt_base, self._answer_prompt_base)
            if any(agent.prompt_exceeds_limit(prompt) for agent, prompt in zip(agents, prompts)):
                return False  # Skip backpropagation due to prompt limit
        # if failed in number of tries, exit
        # reset prompts to base again
        self._question_prompt_base.reset()
        self._answer_prompt_base.reset()
        return False

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
    
    def iterate(
        self,
        node: BTMCTSNode,
        actor: Actor,
        world_model: WorldModel,
        num_children: int,
        sample: dict,
        sample_idx: int,
        max_tries: int
    ) -> Optional[list[BTMCTSNode]]:
        """Runs one iteration of MCTS on the input node using the actor-world model strategy."""
        path = self.select(node)
        # Check if the selected node is terminal or the depth limit is reached
        if self._is_terminal_with_depth_limit(path[-1]):
            cum_reward = self.back_propagate(path)
        else:
            self.expand(path[-1], actor, world_model, num_children, sample, sample_idx)
            success = self.simulate_node(
                path, actor, world_model, max_tries, sample, sample_idx
            )
            if success:
                cum_reward = self.back_propagate(path)
            else:
                return None
        # Update the output based on the specified strategy
        if self.output_strategy == 'max_iter' \
            and path[-1].is_terminal \
            and cum_reward > self._output_cum_reward:
            self._output_cum_reward = cum_reward
            self._output_iter = path
        elif self.output_strategy == 'last_iter':
            self._output_cum_reward = cum_reward
            self._output_iter = path
        elif self.output_strategy == 'last_terminal_iter' and path[-1].is_terminal:
            self._output_cum_reward = cum_reward
            self._output_iter = path

        return path

    def search(self,
               root: BTMCTSNode,
               actor: Actor,
               world_model: WorldModel,
               num_children: int,
               sample: dict,
               sample_idx: int,
               max_tries: int
               ) -> Tuple[list[BTMCTSNode], float]:
        """ Search for the optimal path based on strategy """
        # run mcts for n times and store each explored path
        for _ in trange(self.num_iters, disable=self.use_tqdm, desc='MCTS iteration', leave=False):
            path = self.iterate(root, actor, world_model,
                                num_children, sample, sample_idx, max_tries)
            if path is None:  # if none skip iteration
                message = f'\nError in llm parsing, or reached terminal, skipping MCTS iteration...\n'
                self.logger.info(message) if self.logger else print(message)
                continue
            if self.output_trace_in_each_iter:
                self.trace_in_each_iter.append(deepcopy(path))

        self._output_iter, self._output_cum_reward = SearchStrategies.execute_strategy(root,
                                                                                       self.cum_reward_func,
                                                                                       self.output_strategy)

        return self._output_iter, self._output_cum_reward

    def print_with_optimal(self, root: BTMCTSNode, logger: Optional[logging.Logger] = None):
        console = Console()
        if root is None:
            print("The MCTS tree is empty. Please run the MCTS algorithm first.")
            return
        optimal_path, max_reward = SearchStrategies.execute_strategy(root,
                                                                     self.cum_reward_func,
                                                                     self.output_strategy)
        optimal_node_ids = set(node.id for node in optimal_path)
        rich_tree = self.build_tree(root, optimal_node_ids)
        print(f'Reasoning Trace:')
        print(f'Max Reward: {max_reward}')
        console.print(rich_tree)

    def build_tree(self, node: BTMCTSNode, optimal_node_ids=None):
        if node is None:
            return Tree("[bold red]None[/bold red]")
        parent_id = node.parent.id if node.parent else None
        if node.children:
            children_ids = [child.id for child in node.children]
        else:
            children_ids = []
        node_Q = f"{node.Q:.2f}" if node.Q is not None else "None"
        node_reward = f"{node.reward:.2f}" if node.reward is not None else "None"
        in_optimal_path = optimal_node_ids and node.id in optimal_node_ids
        terminal_color = "green" if node.is_terminal else "red"
        if in_optimal_path:
            node_info = (
                f"[bold red]Node ID:[/] [green]{node.id}[/] | "
                f"[bold red]Parent ID:[/] [magenta]{parent_id}[/] | "
                f"[bold red]Q-Value:[/] [yellow]{node_Q}[/] | "
                f"[bold red]Reward:[/] [yellow]{node_reward}[/] | "
                f"[bold red]Terminal:[/] [{terminal_color}]{node.is_terminal}[/] | "
                f"[bold red]Children IDs:[/] [blue]{children_ids}[/]"
            )
        else:
            node_info = (
                f"[bold cyan]Node ID:[/] [green]{node.id}[/] | "
                f"[bold cyan]Parent ID:[/] [magenta]{parent_id}[/] | "
                f"[bold cyan]Q-Value:[/] [yellow]{node_Q}[/] | "
                f"[bold cyan]Reward:[/] [yellow]{node_reward}[/] | "
                f"[bold red]Terminal:[/] [{terminal_color}]{node.is_terminal}[/] | "
                f"[bold cyan]Children IDs:[/] [blue]{children_ids}[/]"
            )

        rich_tree = Tree(node_info)
        if node.children:
            for child in node.children:
                child_tree = self.build_tree(child, optimal_node_ids)
                rich_tree.add(child_tree)

        return rich_tree

    def guess_answer(self,
                     root: BTMCTSNode,
                     actor: Actor,
                     world_model: WorldModel,
                     num_children: int,
                     sample: dict,
                     sample_idx: int,
                     max_tries: int, 
                     verbose: bool = True
                     ) -> Tuple[str, list[BTMCTSNode]]:
        """ Generates an answer for a gsm8k problem via mcts then inference """  
        # run inference from best current node
        optimal_path, _ = self.search(root, actor, world_model, num_children,
                                        sample, sample_idx, max_tries)
        if verbose: 
            self.print_with_optimal(root)
        # if best leaf is not terminal, then
        if optimal_path[-1].is_terminal:
            answer: str = optimal_path[-1].state.history[-1].content
            return answer, optimal_path
        else:
            self._question_prompt_base.copy_history(optimal_path[-1].state)
            self._answer_prompt_base.copy_history(optimal_path[-1].state)
            
            for _ in range(1, max_tries + 1):
                # generating sub-question
                sub_question = actor.act(self._question_prompt_base)
                self._question_prompt_base.add(role='assistant', content=sub_question)
                self._answer_prompt_base.add(role='user', content=sub_question)
                # generating sub-answer
                step_result = world_model.step_logprobs(self._answer_prompt_base)
                sub_answer = step_result.get('text')
                log_prob = step_result.get('log_probs')
                # updating prompts
                self._question_prompt_base.add(role='user', content=sub_answer)
                self._answer_prompt_base.add(role='assistant', content=sub_answer)
                # using agents and prompts to test if condition below
                agents = (actor, world_model)
                prompts = (self._question_prompt_base, self._answer_prompt_base)
                # if final answer reached or exceed prompt limit, break from loop 
                if filter_output_type(sub_answer) == "final_answer": 
                    break
                elif any(agent.prompt_exceeds_limit(prompt) for agent, prompt in zip(agents, prompts)):
                    break
            self._question_prompt_base.reset()
            self._answer_prompt_base.reset()
            return sub_answer, optimal_path

if __name__ == "__main__":

    from bigtree import find_names
    from rich import print
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

    mcts = MCTS(question_prompt=question_prompt,
                answer_prompt=answer_prompt, num_iters=10)

    generator_cfg = VLLMGeneratorConfig(temperature=0.9)
    generator = VLLMGenerator(generator_cfg)
    actor = Actor(generator)
    world_model = WorldModel(generator)

    # optimal_path, max_reward = mcts.search(
    #     root, actor, world_model, 3, samples[0], 0, 15)
    # mcts.print_with_optimal(root)
    # leaf_node = find_names(root, optimal_path[-1].name)[0]
    # print(leaf_node.state)
    
    answer, optimal_path = mcts.guess_answer(root, actor, world_model, 3, samples[0], 0, 15)
    breakpoint()
    # # name =
    # # find_names(root, str()).state
    # # find_names(root, 'NodeID: 13')[0].state

    # mcts.iterate(root, actor, world_model, 3, samples[0], 0, 10)

    # breakpoint()
