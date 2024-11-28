""" Bigtree version of with LLM MCTS """

from __future__ import annotations
import copy
from typing import Optional, Callable, Any, Literal, Tuple
from rich.console import Console
import numpy as np
from tqdm.rich import trange, tqdm
import random
import math
import logging
# display packages
from rich.tree import Tree
from rich.console import Console
from rich.panel import Panel
from rich.text import Text

# imported classes
from agents.gsm8k.utils import filter_output_type, gsm_is_correct
from agents.mcts.bigtree.bigtree_mcts_node import BTMCTSNode
from agents.reasoners.wm_mcts_reasoner import WorldModel, Actor
from agents.mcts.bigtree.mcts_utils import SearchStrategies
from agents.mcts.bigtree import Prompt, Computable, NodePath
from agents.gsm8k import GSM8KProblem
from agents.prompts.base_prompt_template import BasePromptTemplate
from agents.prompts.llama_prompt import GSMLlamaPromptTemplate


def win_lose(win: bool,
             win_reward: float = 100,
             lose_reward: float = -50
             ) -> float:
    return win_reward if win else lose_reward


class BatchMCTS:
    def __init__(self,
                 question_prompt_base: Prompt,
                 answer_prompt_base: Prompt,
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
        # self._output_iter: list[BTMCTSNode] = None
        # self._output_cum_reward = -math.inf
        self._output_iters: dict[int, NodePath] = {}
        self._output_cum_rewards: dict[int, float] = {}

        self.trace_in_each_iter: dict[int, list[NodePath]] = {}
        self.use_tqdm: bool = use_tqdm
        self.terminal_reward_strategy, self.reward_strategy = reward_strategies.get(
            reward_strategy)

        # base question prompt used for expansion and simulation; should be a blank template
        question_prompt_base.reset(), answer_prompt_base.reset()
        # base answer prompt used for expansion and simulation
        self._question_prompt_base = copy.deepcopy(question_prompt_base)
        self._answer_prompt_base = copy.deepcopy(answer_prompt_base)
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

    def select(self, root: BTMCTSNode) -> list[BTMCTSNode]:
        ''' Goes through start node, and traverses via selecting best uct of children. If no children or terminal or depth-limit hit, return path as is '''
        path = []
        while True:
            path.append(root)
            if self._is_terminal_with_depth_limit(root) or len(root.children) == 0:
                return path
            best_child = self._uct_select(root)
            root = best_child  # set node as best child

    def batch_select(self, roots: list[BTMCTSNode]) -> list[list[BTMCTSNode]]:
        """ Batch selection of multiple best paths for list of roots"""
        paths = [self.select(root)
                 for root in roots]
        return paths

    def batch_expand(self,
                     leaf_nodes: list[BTMCTSNode],
                     actor: Actor,
                     world_model: WorldModel,
                     num_children: int,
                     samples: list[dict],
                     sample_indices: list[int],
                     verbose: bool = False
                     ) -> None:
        """
        Expands a list of nodes into their children and updates the nodes' internal children attributes.
        """
        # Initialize lists to hold prompts and references
        question_prompts: list[Prompt] = []
        answer_prompts: list[Prompt] = []
        node_refs: list[BTMCTSNode] = []
        sample_refs: list[int] = []
        sample_idx_refs: list[int] = []
        node_true_idx_refs: list[int] = []
        node_to_children: dict[int, list] = {
            idx: [] for idx in range(len(leaf_nodes))}

        for idx, node in enumerate(leaf_nodes):
            # Copy prompt history from node state
            self._question_prompt_base.copy_history(node.state)
            self._answer_prompt_base.copy_history(node.state)
            # for each node - copy the prompt base into its children
            for _ in range(num_children):
                # Create copies of the prompts for each child
                question_prompt = copy.deepcopy(self._question_prompt_base)
                answer_prompt = copy.deepcopy(self._answer_prompt_base)
                question_prompts.append(question_prompt)
                answer_prompts.append(answer_prompt)
                # appending node and index references
                node_refs.append(node)
                sample_refs.append(samples[idx])
                sample_idx_refs.append(sample_indices[idx])
                node_true_idx_refs.append(idx)
            # reset it per node so we do not copy multiple questions
            self._question_prompt_base.reset()
            self._answer_prompt_base.reset()

        # NOTE - len(question_prompts) = len(leaf_nodes) * num_children
        # Generate sub_questions in batch
        sub_questions: list[str] = actor.batch_act(question_prompts)
        # fill in the answer_prompts
        # Update answer_prompts with sub_questions
        for idx in range(len(sub_questions)):
            answer_prompts[idx].add(role='user', content=sub_questions[idx])

        # Generate sub_answers with log_probs in batch
        sub_answers = world_model.batch_step_logprobs(answer_prompts)

        # Process each sub_answer and create child nodes
        for idx in range(len(sub_answers)):
            # update each question prompt
            sub_answer = sub_answers[idx]['text']
            log_prob = sub_answers[idx]['log_probs']
            question_prompts[idx].add(
                role='assistant', content=sub_questions[idx])
            question_prompts[idx].add(role='user', content=sub_answer)

            #
            if filter_output_type(sub_answer) == 'final_answer':
                out, message = gsm_is_correct(
                    sample_idx_refs[idx], sub_answer, sample_refs[idx])
                if verbose:
                    print(message)
                reward = self.terminal_reward_strategy(out)
                terminated = True
            else:
                reward = self.reward_strategy(log_prob)
                terminated = False

            child_node = BTMCTSNode(state=copy.deepcopy(question_prompts[idx]),
                                    action=sub_questions[idx],
                                    reward=reward,
                                    parent=node_refs[idx],
                                    is_terminal=terminated,
                                    calc_q=self.calc_q_func)

            node_idx: int = node_true_idx_refs[idx]
            node_to_children[node_idx].append(child_node)

        # assign each leaf node its children
        for idx in node_to_children:
            leaf_nodes[idx].children = node_to_children[idx]

        # Reset the base prompts
        self._question_prompt_base.reset()
        self._answer_prompt_base.reset()

    def batch_simulate_node(self,
                            paths: list[NodePath],
                            actor: Actor,
                            world_model: WorldModel,
                            max_tries: int,
                            samples: list[GSM8KProblem],
                            sample_indices: list[int],
                            verbose: bool = False
                            ) -> list[bool]:
        """ Simulates a single node until end of problem and returns a flag if successfully simulated or not """
        # prompts to simulate
        nodes_to_sim: list[BTMCTSNode] = []
        question_prompts: list[Prompt] = []
        answer_prompts: list[Prompt] = []
        # masks for info for each path
        active_mask: list[bool] = []  # flag for which paths to keep iterating
        # batch stores the return for each paths
        rollout_rewards: list[list[float]] = []
        result_flags: list[bool] = [False] * len(paths)

        for idx, path in enumerate(paths):
            # randomly select child node
            child_idx = random.choice(range(len(path[-1].children)))
            child_node = path[-1].children[child_idx]
            nodes_to_sim.append(child_node)
            # Copy child state for simulation
            question_prompt = copy.deepcopy(self._question_prompt_base)
            answer_prompt = copy.deepcopy(self._answer_prompt_base)
            question_prompt.copy_history(child_node.state)
            answer_prompt.copy_history(child_node.state)
            #
            question_prompts.append(question_prompt)
            answer_prompts.append(answer_prompt)
            rollout_rewards.append([])
            active_mask.append(True)

        for _ in range(1, max_tries + 1):
            # for generating sub_questions
            # Collect indices of active simulations
            active_indices: list[int] = [idx for idx,
                                         active in enumerate(active_mask) if active]
            if not active_indices:
                break  # No active simulations left
            # Prepare prompts for active simulations
            active_question_prompts: list[Prompt] = [
                question_prompts[idx] for idx in active_indices]
            # active_answer_prompts = [answer_prompts[idx] for idx in active_indices]
            # Generate sub-questions in batch
            sub_questions: list[str] = actor.batch_act(active_question_prompts)
            # for generating sub_answers
            # Update question_prompts and answer_prompts with sub_questions
            for idx, i in enumerate(active_indices):
                question_prompts[i].add(
                    role='assistant', content=sub_questions[idx])
                answer_prompts[i].add(role='user', content=sub_questions[idx])
            # Generate sub-answers with log_probs in batch
            active_answer_prompts: list[Prompt] = [
                answer_prompts[i] for i in active_indices]
            step_results: list[dict] = world_model.batch_step_logprobs(
                active_answer_prompts)
            # Process each sub_answer and update prompts
            for idx, i in enumerate(active_indices):
                sub_answer = step_results[idx]['text']
                log_prob = step_results[idx]['log_probs']
                # update prompts with sub_answer
                question_prompts[i].add(role='user', content=sub_answer)
                answer_prompts[i].add(role='assistant', content=sub_answer)
                # Collect reward
                rollout_rewards[i].append(self.reward_strategy(log_prob))

                # Check for termination condition
                if filter_output_type(sub_answer) == 'final_answer':
                    correct, message = gsm_is_correct(
                        sample_indices[i], sub_answer, samples[i])
                    rollout_rewards[i].append(
                        self.terminal_reward_strategy(correct))
                    if verbose:
                        print(message)
                    # Reset prompts to base again
                    question_prompts[i].reset()
                    answer_prompts[i].reset()
                    # Update node reward and path
                    rollout_reward = sum(rollout_rewards[i])
                    nodes_to_sim[i].reward = rollout_reward
                    paths[i].append(nodes_to_sim[i])
                    # Mark as successful and deactivate simulation
                    result_flags[i] = True
                    active_mask[i] = False
                    continue  # Move to the next simulation

                # Check if prompt exceeds limit
                agents = (actor, world_model)
                prompts = (question_prompts[i], answer_prompts[i])
                if any(agent.prompt_exceeds_limit(prompt) for agent, prompt in zip(agents, prompts)):
                    # Deactivate simulation due to prompt limit
                    active_mask[i] = False

        # Reset prompts for all simulations
        for i in range(len(paths)):
            if question_prompts[i] is not None:
                question_prompts[i].reset()
            if answer_prompts[i] is not None:
                answer_prompts[i].reset()

        return result_flags

    def back_propagate(self, path: NodePath) -> float:
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

    def batch_back_propagate(self, paths: list[NodePath]) -> list[float]:
        """ Batch back propagates a list of paths """
        cum_rewards = [self.back_propagate(path)
                       for path in paths]

        return cum_rewards

    def _update_output(self, path: NodePath, cum_reward: float, sample_idx: int) -> None:
        """Updates the output based on the specified strategy for a single sample. """
        current_cum_reward = self._output_cum_rewards.get(sample_idx, -math.inf)

        if self.output_strategy == 'max_iter' and path[-1].is_terminal and cum_reward > current_cum_reward:
            self._output_cum_rewards[sample_idx] = cum_reward
            self._output_iters[sample_idx] = path
        elif self.output_strategy == 'last_iter':
            self._output_cum_rewards[sample_idx] = cum_reward
            self._output_iters[sample_idx] = path
        elif self.output_strategy == 'last_terminal_iter' and path[-1].is_terminal:
            self._output_cum_rewards[sample_idx] = cum_reward
            self._output_iters[sample_idx] = path

    def batch_iterate(
        self,
        roots: list[BTMCTSNode],
        actor: Actor,
        world_model: WorldModel,
        num_children: int,
        samples: list[dict],
        sample_indices: list[int],
        max_tries: int
    ) -> Optional[list[NodePath]]:
        """Runs one iteration of MCTS on batch of input nodes using the actor-world model strategy."""
        paths: list[NodePath] = self.batch_select(roots)
        # Identify terminal and non-terminal paths using term_indices mask - True if terminal
        terminal_indices: list[bool] = [
            self._is_terminal_with_depth_limit(path[-1]) for path in paths]
        # terminal paths that we directly backprop
        terminal_paths: list[NodePath] = [path for path,
                                          is_term in zip(paths, terminal_indices) if is_term]
        # non-terminal paths we need to expand and sim
        sim_paths: list[NodePath] = [path for path, is_term in zip(
            paths, terminal_indices) if not is_term]

        if terminal_paths:
            terminal_sample_indices: list[int] = [idx for idx, is_term in zip(
                sample_indices, terminal_indices) if is_term]
            cum_rewards_terminal: list[float] = self.batch_back_propagate(
                terminal_paths)
            # Update outputs for terminal paths
            for path, cum_reward, sample_idx in zip(terminal_paths, cum_rewards_terminal, terminal_sample_indices):
                self._update_output(path, cum_reward, sample_idx)

        if sim_paths:
            # get sim indices and samples we need for learning - corresponds to the paths we want to expand and sim
            sim_indices: list[int] = [idx for idx, is_term in zip(
                sample_indices, terminal_indices) if not is_term]
            sim_samples: list[GSM8KProblem] = [sample for sample, is_term in zip(
                samples, terminal_indices) if not is_term]
            # get all leaf nodes to expand
            leaves_to_expand: list[BTMCTSNode] = [path[-1]
                                                  for path in sim_paths]
            self.batch_expand(leaves_to_expand, actor, world_model,
                              num_children, sim_samples, sim_indices)

            successes = self.batch_simulate_node(sim_paths, actor, world_model,
                                                 max_tries, sim_samples, sim_indices)
            bp_paths: list[NodePath] = [path for success, path
                                        in zip(successes, sim_paths) if success]
            bp_sample_indices: list[int] = [
                idx for success, idx in zip(successes, sim_indices) if success]

            if bp_paths:
                cum_rewards_bp: list[float] = self.batch_back_propagate(
                    bp_paths)
                # Update outputs for successfully simulated paths
                for path, cum_reward, sample_idx in zip(bp_paths, cum_rewards_bp, bp_sample_indices):
                    self._update_output(path, cum_reward, sample_idx)
            else:
                return None  # assume all sim_paths were not successful
                # Update the output based on the specified strategy
        return paths

    def batch_search(self,
                     roots: list[BTMCTSNode],
                     actor: Actor,
                     world_model: WorldModel,
                     num_children: int,
                     samples: list[dict],
                     sample_indices: list[int],
                     max_tries: int
                     ) -> Tuple[dict[int, NodePath], dict[int, float]]:
        """ Search for the optimal path based on strategy """
        # run mcts for n times and store each explored path
        for _ in trange(self.num_iters, disable=self.use_tqdm, desc='MCTS iteration', leave=False):

            paths: list[NodePath] = self.batch_iterate(roots, actor, world_model,
                                                       num_children, samples, sample_indices, max_tries)
            if paths:
                for idx, path in enumerate(paths):
                    if self.output_trace_in_each_iter:
                        self.trace_in_each_iter[idx].append(
                            (copy.deepcopy(path)))
            else:
                message = f'\nError in llm parsing or reached terminal for all nodes in batch. Skipping MCTS iteration...\n'
                self.logger.info(message) if self.logger else print(message)
                continue
        # print tree for each node
        for idx, root in enumerate(roots):
            self._output_iters[idx], self._output_cum_rewards[idx] = SearchStrategies.execute_strategy(root,
                                                                               self.cum_reward_func,
                                                                               self.output_strategy)
        return self._output_iters, self._output_cum_rewards

    def batch_guess_answer(self,
                           roots: list[BTMCTSNode],
                           actor: Actor,
                           world_model: WorldModel,
                           num_children: int,
                           samples: list[dict],
                           sample_indices: list[int],
                           max_tries: int,
                           verbose: bool = True
                           ) -> Tuple[list[str], list[NodePath], list[Panel]]:
        """ Generates an answer for a gsm8k problem via mcts then inference """
        batch_size: list[int] = len(roots)
        # run inference from best current node
        # note that batch_optimal_paths maps [batch_size] -> NodePath
        batch_optimal_paths, batch_output_cum_rewards = self.batch_search(roots,
                                                                          actor,
                                                                          world_model,
                                                                          num_children,
                                                                          samples,
                                                                          sample_indices,
                                                                          max_tries)
        # Initialize lists to hold the results
        answers: list[str] = [''] * batch_size
        optimal_paths: list[NodePath] = [None] * batch_size
        panels: list[Panel] = [None] * batch_size

        # Lists for batch processing
        question_prompts: list[BasePromptTemplate] = []
        answer_prompts: list[BasePromptTemplate] = []
        idx_to_sample: list[int] = []  # Indices of samples to process
        has_answer: list[bool] = []  # Flags indicating if the sample is done

        # Process each sample
        for idx, root in enumerate(roots):
            optimal_path: NodePath = batch_optimal_paths.get(idx)
            if not optimal_path: 
                raise ValueError(f"Optimal Paths count does not match batch size")
            optimal_paths[idx] = optimal_path
            panel = self.construct_panel(root, sample_indices[idx])
            panels[idx] = panel
            # if leaf state is terminal, then assign leaf state as answer 
            if optimal_path[-1].is_terminal:
                answer: str = optimal_path[-1].state.history[-1].content
                answers[idx] = answer
            # if not terminal, need to run further inference 
            else:
                # Copy history to new prompts
                question_prompt = copy.deepcopy(self._question_prompt_base)
                answer_prompt = copy.deepcopy(self._answer_prompt_base)
                
                question_prompt.copy_history(optimal_path[-1].state)
                answer_prompt.copy_history(optimal_path[-1].state)

                question_prompts.append(question_prompt)
                answer_prompts.append(answer_prompt)
                idx_to_sample.append(idx)
                has_answer.append(False)

        # Batch inference for non-terminal nodes
        for _ in range(1, max_tries + 1):
            # Collect indices of prompts not yet done
            active_indices = [i for i, done in enumerate(has_answer) if not done]
            if not active_indices:
                break 

            # Prepare active prompts
            active_question_prompts: list[Prompt] = [question_prompts[i] for i in active_indices]
            active_answer_prompts: list[Prompt] = [answer_prompts[i] for i in active_indices]
            
            # Generate sub-questions in batch
            sub_questions = actor.batch_act(active_question_prompts)
            # Update prompts with sub-questions
            for idx_active, i in enumerate(active_indices):
                sub_question = sub_questions[idx_active] # get corresponding sub_question
                question_prompts[i].add(role='assistant', content=sub_question)
                answer_prompts[i].add(role='user', content=sub_question)

            # Generate sub-answers with log_probs in batch
            step_results = world_model.batch_step_logprobs(active_answer_prompts)
            # Update prompts with sub-answers and check for termination
            for idx_active, i in enumerate(active_indices):
                step_result = step_results[idx_active]
                sub_answer = step_result.get('text')
                log_prob = step_result.get('log_probs')
                question_prompts[i].add(role='user', content=sub_answer)
                answer_prompts[i].add(role='assistant', content=sub_answer)

                # Check for termination condition
                if filter_output_type(sub_answer) == "final_answer":
                    idx_in_roots = idx_to_sample[i]
                    answers[idx_in_roots] = sub_answer
                    has_answer[i] = True
                elif any(
                    agent.prompt_exceeds_limit(prompt)
                    for agent, prompt in zip(
                        (actor, world_model),
                        (question_prompts[i], answer_prompts[i])
                    )
                ):
                    idx_in_roots = idx_to_sample[i]
                    answers[idx_in_roots] = sub_answer
                    has_answer[i] = True

        # Reset prompts for all
        for question_prompt, answer_prompt in zip(question_prompts, answer_prompts):
            question_prompt.reset(), answer_prompt.reset()
        # Return the answers, optimal paths, and panels
        return answers, optimal_paths, panels

    def construct_panel(self, root: BTMCTSNode, sample_idx: int) -> Panel:
        """ Returns a tree image object with optimal path highlighted """
        # NOTE - potential issue with this
        optimal_path, max_reward = SearchStrategies.execute_strategy(
            root, self.cum_reward_func, self.output_strategy)
        optimal_node_ids = set(node.id for node in optimal_path)
        rich_tree = self.build_tree(root, optimal_node_ids)
        title = Text.assemble(
            f"Sample Question #{sample_idx} - Max Reward = ",
            (f"{max_reward}", "bold red"),
            style="bold purple4"
        )
        panel = Panel(
            rich_tree,
            title=title,
            border_style="white",
            expand=True
        )
        return panel

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
                f"[bold red]Terminal:[/] [bold {terminal_color}]{node.is_terminal}[/] | "
                f"[bold red]Children IDs:[/] [blue]{children_ids}[/]"
            )
        else:
            node_info = (
                f"[bold cyan]Node ID:[/] [green]{node.id}[/] | "
                f"[bold cyan]Parent ID:[/] [magenta]{parent_id}[/] | "
                f"[bold cyan]Q-Value:[/] [yellow]{node_Q}[/] | "
                f"[bold cyan]Reward:[/] [yellow]{node_reward}[/] | "
                f"[bold cyan]Terminal:[/] [{terminal_color}]{node.is_terminal}[/] | "
                f"[bold cyan]Children IDs:[/] [blue]{children_ids}[/]"
            )

        rich_tree = Tree(node_info)
        if node.children:
            for child in node.children:
                child_tree = self.build_tree(child, optimal_node_ids)
                rich_tree.add(child_tree)

        return rich_tree
