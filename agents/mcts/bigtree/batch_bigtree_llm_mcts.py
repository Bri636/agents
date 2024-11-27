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
from agents.reasoners.wm_reasoner import WorldModel, Actor
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
        self._output_iter: list[BTMCTSNode] = None
        self._output_cum_reward = -math.inf
        self.trace_in_each_iter = []
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
        node_to_children = {node: [] for node in leaf_nodes}

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
        breakpoint()
        # Generate sub_answers with log_probs in batch
        sub_answers = world_model.batch_step_logprobs(answer_prompts)

        breakpoint()
        
        # creating batch of prompt bases to work with 
        question_prompt_bases = []
        answer_prompt_bases = []
        for idx, node in enumerate(leaf_nodes): 
            question_prompt_bases.append(copy.deepcopy(self._question_prompt_base))
            answer_prompt_bases.append(copy.deepcopy(self._answer_prompt_base))
            
        
        
        
        
        
        
        # Initialize lists to hold prompts and sub-questions for batching
        action_prompts = []
        node_indices = []
        # make num_children x nodes prompts for batched child_nodes
        for idx, node in enumerate(leaf_nodes):
            # Copy prompt history from node state
            question_prompt: Prompt = copy.deepcopy(self._question_prompt_base)
            question_prompt.copy_history(node.state)
            # generate n child_node prompts for each node
            for _ in range(num_children):
                action_prompts.append(copy.deepcopy(question_prompt))
                node_indices.append(idx)

        # Batch generate sub-questions
        sub_questions: list[str] = actor.batch_act(action_prompts)

        # Prepare answer prompts for batch processing
        answer_prompts = []
        for sub_q, question_prompt in zip(sub_questions, action_prompts):
            # Update the question_prompt with the sub_question
            question_prompt.add(role='assistant', content=sub_q)
            # Prepare answer_prompt
            answer_prompt = copy.deepcopy(self._answer_prompt_base)
            answer_prompt.copy_history(question_prompt)
            answer_prompts.append(answer_prompt)

        # Batch generate sub-answers with log probabilities
        step_results = world_model.batch_step_logprobs(answer_prompts)

        # Now process the results and create child nodes
        node_children = {idx: [] for idx in range(len(leaf_nodes))}  # Initialize

        for i, (node_idx, sub_q, step_result, question_prompt) in enumerate(
            zip(node_indices, sub_questions, step_results, action_prompts)
        ):
            sub_answer = step_result.get('text')
            log_prob = step_result.get('log_probs')

            # Update prompts
            question_prompt.add(role='user', content=sub_answer)

            # Determine if terminal
            sample = samples[node_idx]
            sample_idx = sample_indices[node_idx]
            if filter_output_type(sub_answer) == 'final_answer':
                out, message = gsm_is_correct(sample_idx, sub_answer, sample)
                if verbose:
                    print(message)
                reward = self.terminal_reward_strategy(out)
                terminated = True
            else:
                reward = self.reward_strategy(log_prob)
                terminated = False

            # Create child node
            child_node = BTMCTSNode(
                state=copy.deepcopy(question_prompt),
                action=sub_q,
                reward=reward,
                parent=leaf_nodes[node_idx],
                is_terminal=terminated,
                calc_q=self.calc_q_func
            )

            # Add child node to the corresponding node
            node_children[node_idx].append(child_node)

        # Finally, set the children for each node
        for idx, node in enumerate(leaf_nodes):
            node.children = node_children.get(idx, [])

        # Reset the base prompts
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
        child_idx: int = random.sample(
            range(len(path[-1].children)), 1)[0]  # randomly choose a child
        node_to_sim: BTMCTSNode = path[-1].children[child_idx]
        # copy child state for simulation
        self._question_prompt_base.copy_history(node_to_sim.state)
        self._answer_prompt_base.copy_history(node_to_sim.state)

        rollout_rewards = []
        for _ in range(1, max_tries + 1):
            # generating sub-question
            sub_question = actor.act(self._question_prompt_base)
            self._question_prompt_base.add(
                role='assistant', content=sub_question)
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

    def batch_iterate(
        self,
        roots: list[BTMCTSNode],
        actor: Actor,
        world_model: WorldModel,
        num_children: int,
        samples: list[dict],
        sample_indices: list[int],
        max_tries: int
    ) -> Optional[list[BTMCTSNode]]:
        """Runs one iteration of MCTS on batch of input nodes using the actor-world model strategy."""
        paths: list[NodePath] = self.batch_select(roots)
        # Identify terminal and non-terminal paths using term_indices mask - True if terminal
        terminal_indices: list[bool] = [self._is_terminal_with_depth_limit(path[-1]) for path in paths]
        # terminal paths that we directly backprop
        terminal_paths: list[NodePath] = [path for path, is_term in zip(paths, terminal_indices) if is_term]
        # non-terminal paths we need to expand and sim
        sim_paths: list[NodePath] = [path for path, is_term in zip(paths, terminal_indices) if not is_term]
        if terminal_paths: 
            breakpoint()
            cum_rewards: list[float] = self.batch_back_propagate(terminal_paths)
        if sim_paths: 
            # get sim indices and samples we need for learning - corresponds to the paths we want to expand and sim
            sim_indices: list[int] = [idx for idx, is_term in zip(sample_indices, terminal_indices) if not is_term]
            sim_samples: list[GSM8KProblem] = [sample for sample, is_term in zip(samples, terminal_indices) if not is_term]
            # get all leaf nodes to expand
            leaves_to_expand: list[BTMCTSNode] = [path[-1] for path in sim_paths]
            self.batch_expand(leaves_to_expand, actor, world_model, 
                              num_children, sim_samples, sim_indices) 
            
            breakpoint()
            
        #################
        # Filter terminal and non-terminal paths
        terminal_paths: list[BTMCTSNode] = list(filter(
            lambda path: self._is_terminal_with_depth_limit(path[-1]), paths))
        non_terminal_paths: list[BTMCTSNode] = list(filter(
            lambda path: not self._is_terminal_with_depth_limit(path[-1]), paths))
        # Step 3: Backpropagate terminal paths
        if terminal_paths:
            cum_rewards: list[float] = self.batch_back_propagate(terminal_paths)
        # Step 4: Expand and simulate non-terminal paths
        if non_terminal_paths:
            non_terminal_nodes = [path[-1] for path in non_terminal_paths]
            self.batch_expand(paths[-1], actor, world_model,
                    num_children, samples, sample_indices)
            ### rest of code to be done ###
            
            
        # only backprop paths that are terminal       
        cum_rewards = self.batch_back_propagate() # takes in list[list[BTMCTSNode]]
            
        if self._is_terminal_with_depth_limit(paths[-1]):
            breakpoint()
            cum_reward: list[float] = self.batch_back_propagate(paths)
        else:
            self.batch_expand(paths[-1], actor, world_model,
                              num_children, samples, sample_indices)
            success = self.simulate_node(
                paths, actor, world_model, max_tries, samples, sample_indices
            )
            if success:
                cum_reward = self.back_propagate(paths)
            else:
                return None
        # Update the output based on the specified strategy
        if self.output_strategy == 'max_iter' \
                and paths[-1].is_terminal \
                and cum_reward > self._output_cum_reward:
            self._output_cum_reward = cum_reward
            self._output_iter = paths
        elif self.output_strategy == 'last_iter':
            self._output_cum_reward = cum_reward
            self._output_iter = paths
        elif self.output_strategy == 'last_terminal_iter' and paths[-1].is_terminal:
            self._output_cum_reward = cum_reward
            self._output_iter = paths

        return paths

    def batch_search(self,
                     roots: list[BTMCTSNode],
                     actor: Actor,
                     world_model: WorldModel,
                     num_children: int,
                     samples: list[dict],
                     sample_indices: list[int],
                     max_tries: int
                     ) -> Tuple[list[list[BTMCTSNode]], list[float]]:
        """ Search for the optimal path based on strategy """
        # run mcts for n times and store each explored path
        for _ in trange(self.num_iters, disable=self.use_tqdm, desc='MCTS iteration', leave=False):
            
            path = self.batch_iterate(roots, actor, world_model,
                                      num_children, samples, sample_indices, max_tries)
            
            breakpoint()
            if path is None:  # if none skip iteration
                message = f'\nError in llm parsing, or reached terminal, skipping MCTS iteration...\n'
                self.logger.info(message) if self.logger else print(message)
                continue
            if self.output_trace_in_each_iter:
                self.trace_in_each_iter.append(copy.deepcopy(path))

        self._output_iter, self._output_cum_reward = SearchStrategies.execute_strategy(roots,
                                                                                       self.cum_reward_func,
                                                                                       self.output_strategy)

        return self._output_iter, self._output_cum_reward

    def batch_guess_answer(self,
                           roots: list[BTMCTSNode],
                           actor: Actor,
                           world_model: WorldModel,
                           num_children: int,
                           samples: list[dict],
                           sample_indices: list[int],
                           max_tries: int,
                           verbose: bool = True
                           ) -> Tuple[str, list[BTMCTSNode], Panel]:
        """ Generates an answer for a gsm8k problem via mcts then inference """
        # run inference from best current node
        optimal_paths, _ = self.batch_search(roots,
                                             actor,
                                             world_model,
                                             num_children,
                                             samples,
                                             sample_indices,
                                             max_tries)
        
        breakpoint()
        #################### others ###################

        if verbose:
            panel = self.print_with_optimal(roots)
        # if best leaf is not terminal, then
        if optimal_path[-1].is_terminal:
            answer: str = optimal_path[-1].state.history[-1].content
            return answer, optimal_path, panel
        else:
            self._question_prompt_base.copy_history(optimal_path[-1].state)
            self._answer_prompt_base.copy_history(optimal_path[-1].state)

            for _ in range(1, max_tries + 1):
                # generating sub-question
                sub_question = actor.act(self._question_prompt_base)
                self._question_prompt_base.add(
                    role='assistant', content=sub_question)
                self._answer_prompt_base.add(role='user', content=sub_question)
                # generating sub-answer
                step_result = world_model.step_logprobs(
                    self._answer_prompt_base)
                sub_answer = step_result.get('text')
                log_prob = step_result.get('log_probs')
                # updating prompts
                self._question_prompt_base.add(role='user', content=sub_answer)
                self._answer_prompt_base.add(
                    role='assistant', content=sub_answer)
                # using agents and prompts to test if condition below
                agents = (actor, world_model)
                prompts = (self._question_prompt_base,
                           self._answer_prompt_base)
                # if final answer reached or exceed prompt limit, break from loop
                if filter_output_type(sub_answer) == "final_answer":
                    break
                elif any(agent.prompt_exceeds_limit(prompt) for agent, prompt in zip(agents, prompts)):
                    break
            self._question_prompt_base.reset()
            self._answer_prompt_base.reset()

            return sub_answer, optimal_path, panel
        
        
    def print_with_optimal(self, root: BTMCTSNode) -> Panel:
        optimal_path, max_reward = SearchStrategies.execute_strategy(root,
                                                                     self.cum_reward_func,
                                                                     self.output_strategy)
        optimal_node_ids = set(node.id for node in optimal_path)
        rich_tree = self.build_tree(root, optimal_node_ids)
        title = Text.assemble(
            "Reasoning Trace - Max Reward = ",
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
