from __future__ import annotations

''' Dataclass Container for an Episode <s0, a0, r0, d0, mask_pad, ...>; 
d0 is bool end and mask_pad is for padding obs'''

import torch 
import numpy as np

from typing import Literal, Any
from dataclasses import dataclass
from pydantic import BaseModel
            
class EpisodeMetaData: 
    ''' Metadata container '''
    info: dict[str, Any]
    
    def __len__(self) -> int: 
        return len(self.info)
    
    def __getitem__(self, idx:int) -> dict[str, Any]: 
        return {'info': self.info[idx]}
    
# TODO: MASK IS MEANT TO BE A binary mask TENSOR OF 0 OR 1 THAT IS MEANT TO PAD 
@dataclass
class BatchEpisode: 
    '''Class that contains data for one episode, played until end'''
    observations: torch.Tensor # B, H, W, C
    actions: torch.Tensor # equivalent to bytetensor
    rewards: torch.Tensor
    terminated: torch.Tensor
    masks: torch.Tensor
    
    # use model validate since we want to make the model, then validate the length
    def model_post_init(self) -> None:
        '''We use this to assert that episode len is the same for all data, and truncate at first dones'''
        # assert that data lens are all the same for the episode 
        B = self.observations.shape[0]
        assert all([getattr(self, attr).shape[0]==B for attr in list(self.__dataclass_fields__.keys())]), f'''
        The batch sizes do not match for all fields. 
        '''
        self.observations = self.observations.to(dtype=torch.FloatTensor)
        self.actions = self.actions.to(dtype=torch.LongTensor)
        self.rewards = self.rewards.to(dtype=torch.FloatTensor)
        self.terminated = self.terminated.to(torch.LongTensor)
        self.masks = self.masks.to(torch.BoolTensor)
        
    def __len__(self) -> int: 
        return len(self.observations)
    
    def __getitem__(self, idx:int) -> dict[str, torch.Tensor]: 
        return {
            'observations': self.observations[idx], # shape T, H, W, C)
            'actions': self.actions[idx], # shape T)
            'rewards': self.rewards[idx], # shape T)
            'terminated': self.terminated[idx], # shape T)
            'masks': self.masks[idx] # shape T)
        }
        
@dataclass
class BatchEpisodeMetrics: 
    '''Calculates at the end of a episode'''
    episode_lengths: int
    episode_returns: float
      