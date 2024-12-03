from __future__ import annotations

from dataclasses import dataclass
from typing import Union, Callable
from agents.callbacks.callbacks import BatchMetrics, ThroughputMetrics, ThroughputCallback

Callback = Union[ThroughputCallback]

CallbackMetrics = Union[ThroughputMetrics, BatchMetrics, type]
""" Some type of dataclass """

Registered_Callbacks: dict[str, type | Callable | ThroughputCallback] = {
    'throughput': ThroughputCallback
}