from __future__ import annotations

from dataclasses import dataclass
from typing import List

import numpy as np


@dataclass(frozen=True)
class TaskSpec:
    task_id: str
    source_device: str
    profile: str
    data_size: float
    compute_demand: float
    memory_demand: float
    deadline: float
    priority: int
    privacy_level: int
    max_tx_power: float

    @property
    def sensitivity(self) -> str:
        if self.privacy_level >= 8:
            return "high"
        if self.privacy_level >= 5:
            return "medium"
        return "low"


@dataclass
class NodeSpec:
    node_id: str
    kind: str
    compute_capacity: float
    memory_capacity: float
    io_rate: float
    security_level: int
    base_power: float
    load_amp: float
    reliability: float
    available_bandwidth: float
    queue_backlog: float = 0.0
    cpu_utilization: float = 0.0
    remaining_energy: float = 1.0
    trusted: bool = False


@dataclass(frozen=True)
class Observation:
    vector: np.ndarray
    context: np.ndarray
    feasible_mask: np.ndarray
    current_task: TaskSpec
    candidate_nodes: List[str]


@dataclass(frozen=True)
class ActionDecision:
    route_index: int
    route_name: str
    bandwidth_ratio: float
    cpu_ratio: float
    tx_power: float
    expert_index: int
    expert_name: str
    expert_probs: np.ndarray
    proxy_weights: np.ndarray
