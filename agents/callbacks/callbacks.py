""" Classes and functions for measuring thoroughput """

from __future__ import annotations

from dataclasses import dataclass, field, asdict
import time
from typing import Any, Optional

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
class ThroughputMetrics:
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

    def __init__(self, metrics: ThroughputMetrics = None):
        """
        Initializes the callback with a ThroughputMetrics object.

        Args:
            metrics (ThroughputMetrics, optional): The metrics object to store throughput data.
        """
        self.metrics = metrics or ThroughputMetrics()
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
        self.metrics.samples_per_second = (
            self.metrics.total_samples / self.metrics.total_time
            if self.metrics.total_time > 0 else 0
        )

    def return_metrics(self) -> ThroughputMetrics:
        """Returns the collected throughput metrics."""
        return self.metrics