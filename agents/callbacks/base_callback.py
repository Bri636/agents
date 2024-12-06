""" Base Class for Callbacks """
from __future__ import annotations

from abc import ABC, abstractmethod

class BaseCallback(ABC):
    """
    Callback class to do stuff 
    """
    @abstractmethod
    def on_start(self):
        """
        Called at the start of the training process 
        """
        pass

    @abstractmethod
    def on_batch_start(self):
        """
        Some kind of process called per batch 
        """
        pass

    @abstractmethod
    def on_batch_end(self):
        """
        Some kind of process done per batch 
        """
        pass
    
    def on_epoch_start(self): 
        """
        Optional epoch start method
        """
        pass
    
    def on_epoch_end(self): 
        """
        Optional epoch end method
        """
        pass

    @abstractmethod
    def on_end(self):
        """
        Called at the end of the training process to finalize overall metrics.
        """
        pass

    @abstractmethod
    def return_metrics(self):
        """
        Returns the stored metrics collected over the course of training
        """
        pass