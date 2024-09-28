from __future__ import annotations

'''Q function classes'''

from abc import ABC

from torch import nn, functional as F, Tensor

class DeepQImage(nn.Module): 
    '''Deep Q model --> returns Q values of len action space'''
    
    def __init__(self, 
                 in_channels:int, 
                 out_channels:int, 
                 kernel_size:int, 
                 stride:int,
                 out_features:int,
                 dropout:float
                 ) -> None: 
        super().__init__()
        
        self.in_channels=in_channels
        self.out_channels=out_channels
        self.kernel_size=kernel_size
        self.stride=stride
        self.out_features=out_features
        self.dropout=dropout

        self.conv1=nn.Conv2d(in_channels, out_channels, kernel_size, stride)
        self.maxpool1=nn.MaxPool2d(kernel_size, stride)
        self.act=nn.GELU()
        self.layer1=None
        self.dropout=nn.Dropout(dropout)

    def forward(self, obs:Tensor) -> Tensor: 
        '''Returns Q values tensor of shape action_space'''
        
        x = self.conv1(obs)
        x = self.maxpool1(x)
        _in = x.unsqueeze(0).flatten(start_dim=1).size(1)
        if self.layer1 is None:
            self.layer1 = nn.Linear(_in, self.out_features).to(x.device)
        x = x.unsqueeze(0).flatten(start_dim=1)
        x = self.layer1(x)
        x = self.act(x)
        out = self.dropout(x)
        
        return out

        