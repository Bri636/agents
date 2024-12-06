""" Classes and functions for measuring thoroughput """

from __future__ import annotations

from dataclasses import dataclass, field, asdict
import time, copy
from typing import Any, Optional
from agents.callbacks import BaseCallback

@dataclass
class BatchMetrics:
    """
    Represents metrics for a single batch during training.

    Attributes:
        batch_idx (int): The index of the batch.
        batch_size (int): The number of samples in the batch.
        batch_time (float): The time taken to process the batch in seconds.
        samples_per_second (float): Throughput for the batch in samples per second.
    """
    batch_idx: int
    batch_size: int
    batch_time: float
    samples_per_second: float

@dataclass
class GSMThroughputMetrics:
    """
    Aggregates throughput metrics over the entire training process.

    Attributes:
        total_time (float): Total training time in seconds.
        total_samples (int): Total number of samples processed.
        total_batches (int): Total number of batches processed.
        samples_per_second (float): Average number of samples processed per second across all batches.
        batch_metrics (list[BatchMetrics]): A list of BatchMetrics objects for each batch.
    """
    total_time: float = 0.0
    total_samples: int = 0
    total_batches: int = 0
    samples_per_second: float = 0.0
    batch_metrics: list[BatchMetrics] = field(default_factory=list)

class ThroughputCallback:
    """
    Callback class to measure and store throughput metrics during training.

    Methods:
        on_start():
            Called at the start of the training process.

        on_batch_start(batch_idx: int):
            Called at the start of each batch to initialize timing.

        on_batch_end(batch_idx: int, batch_size: int):
            Called at the end of each batch to calculate and store metrics.

        on_end():
            Called at the end of the training process to finalize overall metrics.

        return_metrics() -> ThroughputMetrics:
            Returns the collected throughput metrics.
    """

    def __init__(self, metrics: GSMThroughputMetrics = None):
        """
        Initializes the callback with a ThroughputMetrics object.

        Args:
            metrics (ThroughputMetrics, optional): The metrics object to store throughput data.
        """
        self.metrics = metrics or GSMThroughputMetrics()
        self._batch_start_time = None
        self._total_start_time = None

    def on_start(self):
        """Called at the start of the training process to initialize total timing."""
        self._total_start_time = time.time()

    def on_batch_start(self, batch_idx: Optional[int]=None):
        """
        Called at the start of each batch to initialize batch timing.

        Args:
            batch_idx (int): The index of the batch.
        """
        self._batch_start_time = time.time()

    def on_batch_end(self, batch_idx: int, batch_size: int):
        """
        Called at the end of each batch to calculate and store batch metrics.

        Args:
            batch_idx (int): The index of the batch.
            batch_size (int): The number of samples in the batch.
        """
        batch_end_time = time.time()
        batch_time = batch_end_time - self._batch_start_time
        samples_per_second = batch_size / batch_time if batch_time > 0 else 0

        # Record batch metrics
        self.metrics.batch_metrics.append(
            BatchMetrics(
                batch_idx=batch_idx,
                batch_size=batch_size,
                batch_time=batch_time,
                samples_per_second=samples_per_second
            )
        )

        # Update totals
        self.metrics.total_samples += batch_size
        self.metrics.total_batches += 1

    def on_end(self):
        """Called at the end of the training process to finalize overall metrics."""
        total_end_time = time.time()
        self.metrics.total_time = total_end_time - self._total_start_time

        # Calculate overall throughput
        self.metrics.seconds_per_sample = (
            self.metrics.total_samples / self.metrics.total_time
            if self.metrics.total_time > 0 else 0
        )

    def return_metrics(self) -> GSMThroughputMetrics:
        """Returns the collected throughput metrics."""
        return copy.deepcopy(self.metrics)
    
    
@dataclass
class GSMBatchMetrics:
    """
    Represents metrics for a single batch during training.

    Attributes:
        batch_idx (int): The index of the batch.
        batch_size (int): The number of samples in the batch.
        batch_time (float): The time taken to process the batch in seconds.
        seconds_per_sample (float): Throughput for the batch in samples per item / question. 
    """
    batch_idx: int
    batch_size: int
    batch_time: float
    seconds_per_sample: float
    
@dataclass
class GSMThroughputMetrics:
    """
    Aggregates throughput metrics over the entire training process.

    Attributes:
        total_time (float): Total training time in seconds.
        total_samples (int): Total number of samples processed.
        total_batches (int): Total number of batches processed.
        samples_per_second (float): Average number of samples processed per second across all batches.
        batch_metrics (list[BatchMetrics]): A list of BatchMetrics objects for each batch.
    """
    total_time: float = 0.0
    total_samples: int = 0
    total_batches: int = 0
    seconds_per_sample: float = 0.0
    batch_metrics: list[GSMBatchMetrics] = field(default_factory=list)
    
class GSMThroughputCallback(BaseCallback): 
    """
    Callback class to measure and store throughput metrics during training.

    Methods:
        on_start():
            Called at the start of the training process.

        on_batch_start(batch_idx: int):
            Called at the start of each batch to initialize timing.

        on_batch_end(batch_idx: int, batch_size: int):
            Called at the end of each batch to calculate and store metrics.

        on_end():
            Called at the end of the training process to finalize overall metrics.

        return_metrics() -> ThroughputMetrics:
            Returns the collected throughput metrics.
    """

    def __init__(self, metrics: GSMThroughputMetrics = None):
        """
        Initializes the callback with a ThroughputMetrics object.

        Args:
            metrics (ThroughputMetrics, optional): The metrics object to store throughput data.
        """
        self.metrics = metrics or GSMThroughputMetrics()
        self._batch_start_time = None
        self._total_start_time = None
        
    def on_start(self) -> None: 
        self._total_start_time: float = time.time()
        
    def on_batch_start(self) -> None:
        """
        Calculates the following metrics per batch: 
        * seconds per batch => implicitly calculates samples per question 
            - batch time 
            - batch size 
            - batch idx 
        """
        self._batch_start_time = time.time()
        
    def on_batch_end(self, batch_idx: int, batch_size: int, num_steps: int = None) -> None:
        """
        Calculates the following metrics per batch: 
        * seconds per batch => implicitly calculates samples per question 
            - batch time 
            - batch size 
            - batch idx 
        """
        batch_end_time: float = time.time()
        batch_time = batch_end_time - self._batch_start_time
        # seconds needed to solve one question 
        seconds_per_sample: float = float(batch_time / batch_size)

        # Record batch metrics
        self.metrics.batch_metrics.append(
            GSMBatchMetrics(
                batch_idx=batch_idx,
                batch_size=batch_size,
                batch_time=batch_time,
                seconds_per_sample=seconds_per_sample
            )
        )

        # Update totals
        self.metrics.total_samples += batch_size
        self.metrics.total_batches += 1

    def on_end(self) -> None:
        """Called at the end of the training process to finalize overall metrics."""
        total_end_time = time.time()
        self.metrics.total_time = total_end_time - self._total_start_time
        # calculate overall samples per second 
        self.metrics.seconds_per_sample = (
            self.metrics.total_time / self.metrics.total_samples
        )
        
    def return_metrics(self) -> GSMThroughputMetrics:
        return copy.deepcopy(self.metrics)
    
    
@dataclass
class MCTSBatchMetrics: 
    batch_idx: int 
    batch_size: int 
    num_steps: int 
    average_steps_per_sample: int 
    
@dataclass
class MCTSMetrics: 
    total_batches: int = 0
    total_number_steps: int = 0
    batch_metrics: list[MCTSBatchMetrics] = field(default_factory=list)
        
class MCTSCallBack(BaseCallback): 
    """ 
    Callback that measures the following: 
    - average number of full steps before termination
    """
    def __init__(self, metrics: MCTSMetrics = None) -> None:
        
        self.metrics = metrics or MCTSMetrics()
        
    def on_start(self) -> None: 
        """ Does Nothing """
        pass
    
    def on_batch_start(self) -> None: 
        """ Does Nothing """
        pass
    
    def on_batch_end(self, batch_idx: int, batch_size: int, num_steps: int) -> None: 
        """ 
        Takes in total number steps for the batch 
        and the batch size to calculate the average number of steps  
        needed for a question 
        """
        average_steps_per_sample = float(num_steps / batch_size)
       
        batch_metrics: MCTSBatchMetrics = MCTSBatchMetrics(
            batch_idx=batch_idx, 
            batch_size=batch_size,
            num_steps=num_steps, 
            average_steps_per_sample=average_steps_per_sample
        )
        
        self.metrics.batch_metrics.append(batch_metrics)
        self.metrics.total_batches += 1
        self.metrics.total_number_steps += num_steps
        
    def on_end(self):
        """ Do Nothing """
        pass
    
    def return_metrics(self) -> MCTSMetrics:
        return copy.deepcopy(self.metrics)