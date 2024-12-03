from __future__ import annotations

""" Tests with MCTS Node """
import pickle
from os import PathLike
import pickle
import math
from copy import deepcopy
import copy
from typing import Generic, Optional, NamedTuple, Callable, Hashable, Any, Literal, Tuple, Union
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
from tqdm import tqdm

from agents.algorithms.tree_search.base import (SearchAlgorithm, WorldModel, SearchConfig,
                                                State, Action, Example, Trace)
from agents.utils import calculate_returns
# from agents.algorithms.tree_search.mcts_simple import MCTS, MCTSNode
from agents.gsm8k.utils import filter_output_type, gsm_is_correct
from agents.algorithms.tree_search.mcts_node import MCTSNode
from agents.reasoners.wm_reasoner import WorldModel, Actor

from agents.prompts.base_prompt_template import BasePromptTemplate
from agents.prompts.llama_prompt import GSMLlamaPromptTemplate

class MCTSResult(NamedTuple):
    """ Simple Container Class for MCTS Output """

    terminal_state: State
    cum_reward: float
    trace: Trace
    trace_of_nodes: list[MCTSNode]
    tree_state: MCTSNode
    trace_in_each_iter: list[list[MCTSNode]] = None
    tree_state_after_each_iter: list[MCTSNode] = None
    aggregated_result: Optional[Hashable] = None

class MCTS(SearchAlgorithm, Generic[State, Action, Example]):
    def __init__(self,
                 output_trace_in_each_iter: bool = False,
                 w_exp: float = 1.,
                 depth_limit: int = 5,
                 num_iters: int = 10,
                 cum_reward: Callable[[list[float]], float] = sum,
                 calc_q: Callable[[list[float]], float] = np.mean,
                 simulate_strategy: str | Callable[[list[float]], int] = 'max',
                 output_strategy: str = 'max_reward',
                 use_tqdm: bool = True,
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

        # self.root: MCTSNode = root

        # self.world_model = None
        # self.search_config = None
        self.output_trace_in_each_iter = output_trace_in_each_iter
        self.w_exp = w_exp
        self.depth_limit = depth_limit
        self.num_iters = num_iters
        self.cum_reward = cum_reward
        self.calc_q = calc_q
        assert output_strategy in ['max_reward', 'follow_max',
                                   'max_visit', 'max_iter',
                                   'last_iter', 'last_terminal_iter']
        self.output_strategy = output_strategy
        self._output_iter: list[MCTSNode] = None
        self._output_cum_reward = -math.inf
        self.trace_in_each_iter: list[list[MCTSNode]] = None
        self.use_tqdm = use_tqdm
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
               num_children: int,
               sample: dict,
               win_reward: int = 100,
               lose_reward: int = -10
               ) -> None:
        """ Expands last node of path into d children and updates the nodes internal children attribute """
        question_prompt = copy.deepcopy(question_prompt)
        answer_prompt = copy.deepcopy(answer_prompt)
        # current prompt that encodes the history of qa interactions
        state: GSMLlamaPromptTemplate = copy.deepcopy(node.state)
        question_prompt.copy_history(state)
        answer_prompt.copy_history(state)
        # sub_questions = [actor.act(state) for _ in range(num_children)]
        sub_questions_pkg: list[dict[str, str | list[str] | list[float]]] = [actor.act_logprobs(state)
                                                                             for _ in range(num_children)]
        sub_questions = [sub_question['text']
                         for sub_question in sub_questions_pkg]

        # # NOTE: THIS MUST BE LOG PROBS FOR ANSWER
        # log_probs = [sub_question['log_probs'] for sub_question in sub_questions_pkg]

        children = []
        for sub_question in sub_questions:

            answer_prompt.add(**{'role': 'user', 'content': sub_question})
            sub_answer, log_prob = world_model.step_logprobs(
                answer_prompt).values()
            question_prompt.add(
                **{'role': 'assistant', 'content': sub_question})
            question_prompt.add(**{'role': 'user', 'content': sub_answer})

            if filter_output_type(sub_answer) == 'final_answer':
                terminated = True
                out = gsm_is_correct(0, sub_answer, sample)
                if out:
                    reward = win_reward
                else:
                    reward = lose_reward
            else:
                reward = np.mean(log_prob)
                terminated = False  # set as not done
            child_node = MCTSNode(state=copy.deepcopy(question_prompt),
                                  action=sub_question,
                                  reward=reward,
                                  parent=node,
                                  is_terminal=terminated,
                                  calc_q=self.calc_q
                                  )
            answer_prompt.pop()
            question_prompt.pop([-1, -2])
            children.append(child_node)

        node.children = children

    def simulate_node(self,
                      path: list[MCTSNode],
                      actor: Actor,
                      world_model: WorldModel,
                      max_tries: int,
                      sample: dict,
                      question_prompt: GSMLlamaPromptTemplate,
                      answer_prompt: GSMLlamaPromptTemplate,
                      win_reward: float = 100,
                      lose_reward: float = -10
                      ) -> bool:
        """ Simulates a single node until end of problem """
        def episode_stop_condition(step: int, terminated: bool, prompt_exceeds_limit: list[bool]) -> bool:
            """ True if max_tries exceeded or terminal"""
            return bool(step > max_tries or terminated or prompt_exceeds_limit)

        # NOTE - these should be just blank fsl prompts
        new_question_prompt = GSMLlamaPromptTemplate(
            **question_prompt.prompt_kwargs)
        new_answer_prompt = GSMLlamaPromptTemplate(
            **answer_prompt.prompt_kwargs)

        child_idx: int = random.sample(
            range(len(path[-1].children)), 1)[0]  # randomly choose a child
        node_to_sim: MCTSNode = path[-1].children[child_idx]
        state: GSMLlamaPromptTemplate = copy.deepcopy(node_to_sim.state)

        new_question_prompt.copy_history(state)
        new_answer_prompt.copy_history(state)
        
        step = 0
        terminated = False
        exceeds_limit = False
        rewards = []
        # if prompt limit exceeded, stop simulation
        while not episode_stop_condition(step, terminated, exceeds_limit):

            sub_question = actor.act(new_question_prompt)
            new_question_prompt.add(
                **{'role': 'assistant', 'content': sub_question})
            new_answer_prompt.add(**{'role': 'user', 'content': sub_question})

            sub_answer, log_prob = world_model.step_logprobs(
                new_answer_prompt).values()
            new_question_prompt.add('user', sub_answer)
            new_answer_prompt.add('assistant', sub_answer)
            # NOTE: if correct, give big reward
            rewards.append(np.mean(log_prob))
            if filter_output_type(sub_answer) == 'final_answer':
                out = gsm_is_correct(0, sub_answer, sample)
                if out:
                    rewards.append(win_reward)
                else:
                    rewards.append(lose_reward)
                terminated = True
            exceeds_limit: list[bool] = any([agent.prompt_exceeds_limit(prompt)
                                            for agent, prompt in zip(
                                            (actor, world_model), (question_prompt, answer_prompt)
                                            )
                                            ])
            step += 1
        # NOTE - actually maybe not problem; suppose you go off the rails and it is stored. that path is never taken anyways.
        # TODO: what happens if you just get stuck in a loop? - you cant backpropagate
        if step > max_tries or exceeds_limit:
            return False  # flag for skipping backprop; assumption - parsing error
        # idea return flag, if flag then skip backpropagation and then move to next
        rollout_reward = sum(rewards)
        node = path[-1].children[child_idx]
        node.reward = rollout_reward
        path.append(node)

        return True

    def back_propagate(self, path: list[MCTSNode]) -> float:
        """ 
        Updates each node in the path with the cumulative rewards from rollout and returns the updated path and the cum_reward for the root 

        ex. leaf node gets rollout reward 
        leaf node - 1 gets rollout reward + own reward 
        leaf node - 2 gets rollout reward + leaf node -1 + own reward 
        ...

        :param path - list[MCTSNode]: list of nodes corresponding to search path 
        :param child_idx - int: Inside the leaf node, the idx of its expanded child node we simulated
        """
        cum_reward_func: Callable[[
            list[float]], float] = self.cum_reward  # way to calculate the cumulative reward from each step. Defaults: sum
        rewards = []  # holds rewards for each node
        cum_reward = -math.inf
        for node in reversed(path):  # leaf --> root
            # ex. leaf: rewards = [100]; leaf-1: rewards = [100, 10]; leaf-2: rewards = [100, 10, 15], ...
            rewards.append(node.reward)
            # NOTE: work-around for node.reward = None => we filter this out
            rewards = list(filter(lambda x: x != None, rewards))
            # self.cum_rewards callable sum; ex. sum([10, 100]), sum([15, 10, 100])
            cum_reward = cum_reward_func(rewards[::-1])
            # node.cum_rewards stores summed rewards for one iteration; ex. (leaf-1).cum_reward = 110
            node.cum_rewards.append(cum_reward)

        return cum_reward

    def iterate(self,
                node: MCTSNode,
                actor: Actor,
                world_model: WorldModel,
                question_prompt: GSMLlamaPromptTemplate,
                answer_prompt: GSMLlamaPromptTemplate,
                num_children: int,
                sample: dict,
                max_tries: int,
                win_reward: float = 50,
                lose_reward: float = 10
                ) -> Union[None, list[MCTSNode]]:
        """ Performs one MCTS iteration on an input node: for now, we set to root node """
        simulated = False
        # collect container blank prompts for node simulation
        path = self.select(node)
        # TEST: get current cumulative rewards
        cum_reward = path[-1].cum_rewards 
        # if leaf is not terminal or exceeds depth
        if not self._is_terminal_with_depth_limit(path[-1]):
            self.expand(path[-1], actor, world_model, question_prompt,
                        answer_prompt, num_children, sample, win_reward, lose_reward)
            simulated = self.simulate_node(
                path, actor, world_model, max_tries, sample, question_prompt, answer_prompt)  # simulate the path
            
        if simulated:
            cum_reward = self.back_propagate(path) # if no parsing errors, backprop
            # set attr to cumulative reward
            self._output_cum_reward = cum_reward
            # if last node is terminal and finds the path with the best cum_reward
            if self.output_strategy == 'max_iter' and path[-1].is_terminal and cum_reward > self._output_cum_reward:
                self._output_iter = path
            # only returns the last iteration
            elif self.output_strategy == 'last_iter':
                self._output_iter = path
            # only returns the last iteration where the leaf node is terminal
            elif self.output_strategy == 'last_terminal_iter' and path[-1].is_terminal:
                self._output_iter = path

            return path
        
        # NOTE: since simulate and backprop both add the selected child node to path, return None if cant simulate properly 
        # aka not worth the trouble
        else: 
            return None # else, return no path

    def search(self,
               root: MCTSNode,
               actor: Actor,
               world_model: WorldModel,
               question_prompt: GSMLlamaPromptTemplate,
               answer_prompt: GSMLlamaPromptTemplate,
               num_children: int,
               sample: dict,
               max_tries: int,
               win_reward: float = 100,
               lose_reward: float = 10
               ):
        """ Search for the optimal path based on strategy """
        # stores the
        self._output_cum_reward = -math.inf
        self._output_iter = None

        # stores the paths in each mcts iteration
        if self.output_trace_in_each_iter:
            self.trace_in_each_iter = []

        # run mcts for n times and store each explored path
        for _ in trange(self.num_iters, disable=self.use_tqdm, desc='MCTS iteration', leave=False):
            path = self.iterate(root,
                                actor,
                                world_model,
                                question_prompt,
                                answer_prompt,
                                num_children,
                                sample,
                                max_tries,
                                win_reward,
                                lose_reward)
            if path is None: 
                print(f'\nError in LLM parsing, skipping MCTS iteration...\n')
                continue
            if self.output_trace_in_each_iter:
                self.trace_in_each_iter.append(deepcopy(path))

        # two different output strategies after running MCTS - this is inference
        # Strategy 1: get path that maximizes reward - greedy
        if self.output_strategy == 'follow_max':
            self._output_iter = []  # stores nodes in path
            cur = root  # make node
            while True:
                self._output_iter.append(cur)  # add to path iteratively
                if cur.is_terminal:  # return path if terminal node
                    break
                # get the children from node if child.state is not None
                visited_children = [
                    x for x in cur.children if x.state is not None]
                # if node has no children, then return path self._output_iter
                if len(visited_children) == 0:
                    break
                # else, get the max child based on its reward
                cur = max(visited_children, key=lambda x: x.reward)
            # set the output_cum_reward as the sum of the node rewards in the output iter trace
            # NOTE: OG had self._output_iter[1::-1] but this takes idx [1, 0]
            self._output_cum_reward = self.cum_reward(
                [node.reward for node in self._output_iter[::-1]])

        # Strategy 2: get the absolute max reward of the inference path
        if self.output_strategy == 'max_reward':
            # use dfs to get the inference path and cum_reward
            self._output_cum_reward, self._output_iter = self.dfs_max_reward(
                root)
            if self._output_cum_reward == -math.inf:
                self._output_iter = None

    def execute(self,
                root: MCTSNode,
                actor: Actor,
                world_model: WorldModel,
                question_prompt: GSMLlamaPromptTemplate,
                answer_prompt: GSMLlamaPromptTemplate,
                num_children: int,
                sample: dict,
                win_reward: float = 100,
                lose_reward: float = 10
                ):

        self.search(root, actor, world_model, question_prompt,
                    answer_prompt, num_children, sample, win_reward, lose_reward)
        if self._output_iter is None:
            terminal_state = trace = None
        else:
            terminal_state = self._output_iter[-1].state
            trace = [node.state for node in self._output_iter], [
                node.action for node in self._output_iter[1:]]
        if self.output_trace_in_each_iter:
            trace_in_each_iter = self.trace_in_each_iter
            tree_state_after_each_iter = [trace[0]
                                          for trace in trace_in_each_iter]
        else:
            trace_in_each_iter = tree_state_after_each_iter = None

        result = MCTSResult(terminal_state=terminal_state,
                            cum_reward=self._output_cum_reward,
                            trace=trace,
                            trace_of_nodes=self._output_iter,
                            tree_state=root,
                            trace_in_each_iter=trace_in_each_iter,
                            tree_state_after_each_iter=tree_state_after_each_iter)

        return result

    def dfs_max_reward(self, path: list[MCTSNode] | MCTSNode) -> tuple[float, list[MCTSNode]]:
        """ Recursively searches for path that maximizes total reward over the path """
        if isinstance(path, MCTSNode):
            path = [path]

        cur = path[-1]
        if not cur.children:
            # Leaf node (no children)
            cumulative_reward = self.cum_reward(
                [node.reward for node in path[1:]])
            return cumulative_reward, path
        else:
            max_reward = -math.inf
            best_path = path
            for child in cur.children:
                reward, child_path = self.dfs_max_reward(path + [child])
                if reward > max_reward:
                    max_reward = reward
                    best_path = child_path
                    
            return max_reward, best_path

    def print_with_optimal(self, root: MCTSNode):
        console = Console()
        if root is None:
            print("The MCTS tree is empty. Please run the MCTS algorithm first.")
            return
        # Get the optimal path
        max_reward, optimal_path = self.dfs_max_reward([root])
        # Get the set of node IDs in the optimal path
        optimal_node_ids = set(node.id for node in optimal_path)
        # Build the tree, passing in the optimal_node_ids
        rich_tree = self.build_tree(root, optimal_node_ids)
        console.print(rich_tree)
        print(f"Maximum Cumulative Reward: {max_reward}")

    def build_tree(self, node, optimal_node_ids=None):
        # Handle the case where node is None
        if node is None:
            return Tree("[bold red]None[/bold red]")

        # Get parent ID, handle None
        parent_id = node.parent.id if node.parent else None

        # Get children IDs, handle empty list or None
        if node.children:
            children_ids = [child.id for child in node.children]
        else:
            children_ids = []

        # Safely format Q-value and reward, handling None
        node_Q = f"{node.Q:.2f}" if node.Q is not None else "None"
        node_reward = f"{node.reward:.2f}" if node.reward is not None else "None"

        # Check if node is in the optimal path
        in_optimal_path = optimal_node_ids and node.id in optimal_node_ids
        terminal_color = "green" if node.is_terminal else "red"
        # Customize the node information with colors
        if in_optimal_path:
            # Color the node info red if it's in the optimal path
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

        # Create the tree node with the colored node_info
        rich_tree = Tree(node_info)

        # Recurse for each child
        if node.children:
            for child in node.children:
                child_tree = self.build_tree(child, optimal_node_ids)
                rich_tree.add(child_tree)

        return rich_tree


if __name__ == "__main__":
    from agents.reasoners.wm_reasoner import Actor, WorldModel
    from agents.gsm8k.utils import read_jsonl_dataset, batch_sample_gsm
    from agents.prompts.llama_prompt import GSMLlamaPromptTemplate
    from agents.generators.vllm_generator import VLLMGenerator, VLLMGeneratorConfig

    data_path = '/lus/eagle/projects/FoundEpidem/bhsu/2024_research/agents/agents/data/gsm.jsonl'
    batch_size = 16

    dataset = read_jsonl_dataset(data_path)
    samples = batch_sample_gsm(dataset, batch_size)

    question_prompt: GSMLlamaPromptTemplate = GSMLlamaPromptTemplate(
        'question', 1, 'question')
    answer_prompt: GSMLlamaPromptTemplate = GSMLlamaPromptTemplate(
        'answer', 1, 'answer')

    question = samples[0]['question']
    question_prompt.add('user', content=question)
    answer_prompt.add('user', content=question)

    root = MCTSNode(state=question_prompt,  # state is original question
                    action=None,
                    reward=None,
                    parent=None,
                    is_terminal=False
                    )

    mcts = MCTS(use_tqdm=False, num_iters=25)

    generator_cfg = VLLMGeneratorConfig(temperature=0.9)
    generator = VLLMGenerator(generator_cfg)
    actor = Actor(generator)
    world_model = WorldModel(generator)

    result = mcts.execute(root, actor, world_model, question_prompt,
                          answer_prompt, num_children=3, sample=samples[0])
    
    mcts.print_with_optimal(root)

    breakpoint()

    dummy_question_prompt: GSMLlamaPromptTemplate = GSMLlamaPromptTemplate(
        'question', 1, 'question')
    dummy_answer_prompt: GSMLlamaPromptTemplate = GSMLlamaPromptTemplate(
        'answer', 1, 'answer')

    for i in range(3):
        # select a path from root node down to leaf node or not that is not fully expanded
        path = mcts.select(mcts.root)
        # if last node is not terminal
        if not mcts._is_terminal_with_depth_limit(path[-1]):
            # expand on last node --> make all of the children
            mcts.expand(path[-1], actor, world_model,
                        question_prompt, answer_prompt, 3, samples[0])
            simulated = mcts.simulate_node(
                path, actor, world_model, 10, samples[0], dummy_question_prompt, dummy_answer_prompt)  # simulate the path
            if simulated:  # only updated if successfully rolled out
                cum_reward = mcts.back_propagate(path)

    mcts.print_with_optimal()

    max_reward, best_path = mcts.dfs_max_reward([mcts.root])

    breakpoint()
