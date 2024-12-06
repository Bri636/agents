from __future__ import annotations

from dataclasses import dataclass
from typing import Union, Callable
from agents.callbacks.base_callback import BaseCallback
from agents.callbacks.callbacks import (GSMBatchMetrics, GSMThroughputMetrics, GSMThroughputCallback, 
                                        MCTSBatchMetrics, MCTSMetrics, MCTSCallBack)

Callback = Union[BaseCallback, GSMThroughputCallback, MCTSCallBack]

CallbackMetrics = Union[GSMThroughputMetrics, MCTSMetrics, type]
""" Some type of dataclass """

Registered_Callbacks: dict[str, type | Callable | GSMThroughputMetrics | MCTSCallBack] = {
    'throughput': GSMThroughputCallback, 
    'mcts': MCTSCallBack
}