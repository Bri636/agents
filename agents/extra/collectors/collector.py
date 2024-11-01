from __future__ import annotations

''' Collects data from a rl environment '''
import gymnasium as gym
from gymnasium import Wrapper
from gymnasium.vector import AsyncVectorEnv

from einops import rearrange
from typing import Union, Any, Callable, Sequence, TypeVar, Optional
from typing_extensions import Self
import numpy as np
import torch
from torch import Tensor
from tqdm import tqdm
from pydantic import field_validator, model_validator

from agents.utils import BaseConfig, calculate_returns
from agents.episodes.episode import BatchEpisode, BatchEpisodeMetrics
from agents.llm_agents.strategist import LactChainStrategyChain

def sample_stop_condition(step: int,
                          max_steps_allowed: int,
                          ratio_env_completed: float,
                          complete_ratio: float
                          ) -> bool:
    '''Bool Callable that determines when we stop sampling
    
    Inputs: 
    ======
    step: int 
        The current step for the vector_env 
    max_steps_allowed: int 
        The maximum steps allowed for the vector_env 
    ratio_env_completed: float 
        The current ratio of environments in the vector_env that have terminated as least once 
    complete_ratio: float 
        The ratio that we terminate at with respect to the number of environments terminated at least once
    
    Output: 
    ======
    Stop: bool 
        Whether to reset the vector env process or not 
    '''

    if (step > max_steps_allowed) or (ratio_env_completed > complete_ratio):
        return True
    else:
        return False

class CollectorSampleConfig(BaseConfig): 
    ''' Kwargs for sample_episode '''
    max_num_steps: int = 10000 # max number of steps before we terminate
    num_episodes: int = 128  # num total episodes to sample
    batch_size: int = 32 # 
    complete_ratio: float = 0.8  # ratio of completed environments that we can reset
    burn_in: bool = False
    epsilon: float = 0.99
    temperature: float = 0.01
    random_policy: bool = True
    seed: int = 1000
    
    @model_validator(mode='after')
    def check_batch_size(self) -> Self: 
        num_episodes = self.num_episodes
        batch_size = self.batch_size
        if num_episodes % batch_size != 0: 
            raise ValueError(f'''
                             Error: num_episodes {num_episodes} must be divisible % by batch_size {batch_size}
                             aka the number of environments in vector_env
                             ''')
        return self
    
class Collector:
    ''' Collects a batch of episodes from an environment '''

    def __init__(self,
                 strategy: LactChainStrategyChain, 
                 actor: ActorAgent, 
                 environment:Union[gym.Env, AsyncVectorEnv],
                 environment_wrappers: Optional[Sequence[Wrapper]]=None
                 ) -> None:
        
        self.actor = actor
        self.environment = environment
        self.environment_wrappers = environment_wrappers

    def sample_episode(self,
                       max_num_steps: int,  # max number of steps before we terminate
                       num_episodes: int,  # proxy for batch size
                       batch_size: int,
                       complete_ratio: float,  # ratio of completed environments that we can reset
                       burn_in: bool = False,
                       epsilon: float = 0.99,
                       temperature: float = 0.01,
                       random_policy: bool = True,
                       seed: int = 1000,
                       **kwargs
                       ) -> BatchEpisode:
        ''' 
        Samples a batch of episodes from the environment and returns a list of episodes

        Num_episodes == num_envs == "batch_size"
        '''
        
        episode_data = {
            'observations': [],
            'actions': [],
            'rewards': [],
            'terminated': [],
            'masks': [],
        }
        episode_metadata = {
            'start_mask': [],
            'info': []
        }

        # (num_envs,) => stores the position of first Terminated for each env
        first_terminate_pos = np.full(batch_size, -1 * np.inf)
    
        step = 0
        episodes_completed = 0
        ratio_episodes_completed = episodes_completed // num_episodes
        terminated = False
        obs, info = self.environment.reset()  # either shape (NUM_ENV, H, W, C) or (H, W, C)

        #  progress bar
        episodes_pb = tqdm(
            iterable=range(0, int(batch_size*complete_ratio)),
            desc=f'Collecting a minimum of {batch_size} episodes aka episode batch size',
            initial=0,
            leave=True,
        )
        
        while not sample_stop_condition(step, 
                                        max_num_steps-1, 
                                        ratio_episodes_completed, 
                                        complete_ratio
                                        ):

            episode_data['observations'].append(obs)
            action = self.actor.act(obs)
            next_obs, reward, terminated, _, info = self.environment.step(
                action)  # reward => (num_envs, )

            obs = next_obs
            episode_data['actions'].append(action)
            episode_data['rewards'].append(reward)
            episode_data['terminated'].append(terminated)
            episode_metadata['info'].append(info)
            # if any of the terminated contains True, set the env in mask_pos to current step
            if np.any(terminated):
                for idx in range(len(terminated)):
                    if terminated[idx] and first_terminate_pos[idx] == -1 * np.inf:
                        first_terminate_pos[idx] = step
                        episodes_completed += 1
                        ratio_episodes_completed = episodes_completed / num_episodes
                        print(ratio_episodes_completed)
                        episodes_pb.update(1)
            step += 1

            # steps_pb.update(1)

        _, _ = self.environment.reset()
        
        if isinstance(self.environment, gym.vector.AsyncVectorEnv):
            episode_data: dict[str, Tensor] = {k: torch.from_numpy(np.array(v)).transpose(0, 1)
                                               for k, v in episode_data.items() if k != 'masks'}

        # TODO: figure out the by default this is the max episode length = highest in 
        max_episode_length = episode_data["actions"].shape[1]
        episode_data['masks'] = self.pad_episodes(
            first_terminate_pos, max_episode_length)
        episode_data['observations'] = rearrange(
            episode_data['observations'], 'B T H W C -> B T C H W')

        return BatchEpisode(**episode_data)
    
    def pad_episodes(self, mask_pos: np.ndarray, max_episode_length: int) -> torch.Tensor:
        ''' Pads each episode in a batch with 0s starting from when the episode terminated '''

        # Replace -inf with max_episode_length in a vectorized manner
        mask_pos_trunc = np.where(mask_pos == -1 * np.inf, max_episode_length, mask_pos).astype(int)
        
        # Convert to PyTorch tensor directly
        mask_pos_trunc = torch.from_numpy(mask_pos_trunc).long()

        # Create a tensor for all zeros, and we'll fill it with ones where needed
        padded_masks = torch.zeros((mask_pos_trunc.size(0), max_episode_length), dtype=torch.float32)

        # Fill in the ones up to mask_pos_trunc for each episode in a vectorized manner
        for i, step in enumerate(mask_pos_trunc):
            padded_masks[i, :step] = 1  # Set ones up to the step position

        return padded_masks

    def calculate_batch_metrics(self, batch_episode: BatchEpisode, gamma: float) -> BatchEpisodeMetrics:
        ''' Calculates the metrics for a batch of episodes '''
        metrics = {}
        B, T = batch_episode.rewards.shape
        batch_rewards = batch_episode.rewards  # B, T
        batch_length = [T] * B

        metrics['episode_returns'] = calculate_returns(batch_rewards, gamma)
        metrics['episode_lengths'] = torch.tensor(batch_length)

        return BatchEpisodeMetrics(**metrics)
    
    def environment_shutdown(self) -> None: 
        self.environment.close()

    def log_metrics(self) -> dict[str, Any]:
        return ...


