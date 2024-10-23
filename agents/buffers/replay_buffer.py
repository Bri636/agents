from __future__ import annotations

""" Dataset Class """
import os
from torch.utils.data import Dataset
from typing import Any, Tuple, Union, Callable, Optional
from pathlib import Path
import numpy as np
import torch
from tensordict import TensorDict
from collections import deque
from dataclasses import asdict
from tqdm import tqdm
from datasets import Dataset as HFDataset, load_from_disk, concatenate_datasets
from torchvision.transforms.functional import to_pil_image
import random
from einops import rearrange

from agents.episodes.episode import BatchEpisode, BatchEpisodeMetrics

PathLike = Union[str, Path]

class ReplayBuffer:
    """Replay Buffer for storing past experiences allowing the agent to learn from them.
    Pulled from: https://lightning.ai/docs/pytorch/stable/notebooks/lightning_examples/reinforce-learning-DQN.html

    Args:
        capacity: size of the buffer

    """

    def __init__(self,
                 name: str,
                 max_num_episodes: int,
                 num_episodes: int,
                 batch_size: int
                 ) -> None:

        self.name = name if name is not None else 'dataset'
        self.num_episodes = num_episodes
        self.max_num_episodes = max_num_episodes  # total possible episode
        self.batch_size = batch_size  # batch size aka num envs in vector_env
        self.num_seen_episodes = 0  # num of episodes in episodes

        # containers
        self.episodes_dict = TensorDict({}, batch_size=[self.batch_size])
        self.episodes = deque()
        # maps episode id aka num episode to the index of the deque
        self.episode_id_to_queue_idx = dict()
        self.newly_modified_episodes = set()
        self.newly_deleted_episodes = set()

    def __len__(self) -> int:
        ''' Returns number episodes '''
        return len(self.episodes)

    def add_batch_episodes(self, episodes: BatchEpisode) -> None:
        ''' Adds a batch of episodes to the buffer '''
        if list(self.episodes_dict.keys()):
            self.episodes_dict = torch.cat([self.episodes_dict,
                                            TensorDict(asdict(episodes),
                                                       batch_size=[
                                                           self.batch_size]
                                                       )
                                            ], dim=0)
        else:
            self.episodes_dict.update(TensorDict(asdict(episodes),
                                                 batch_size=[self.batch_size]
                                                 ))  # add to tensor dict

        self.num_seen_episodes += 1
        for episode in self.episodes:
            self.episodes.append(episode)
        self.episode_id_to_queue_idx[self.num_seen_episodes] = len(
            self.episodes)

    def write_to_hfdataset(self, dir_path: str, num_proc: Optional[int] = None) -> None:
        ''' Writes current buffer to dataset at dir_path '''
        from uuid import uuid4
        from datasets import Dataset as HFDataset

        dataset_dir: PathLike = Path(dir_path) / f'{uuid4()}'

        if not os.path.exists(dataset_dir):
            print(f'Making directory at {dir_path}')
            os.makedirs(dataset_dir, exist_ok=True)

        dataset = HFDataset.from_dict(self.episodes_dict)
        HFDataset.save_to_disk(dataset, dataset_dir, num_proc=num_proc)