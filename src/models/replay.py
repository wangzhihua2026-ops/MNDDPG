from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Tuple

import numpy as np


@dataclass(frozen=True)
class Transition:
    obs: np.ndarray
    context: np.ndarray
    mask: np.ndarray
    seq_obs: np.ndarray
    action_route: np.ndarray
    action_continuous: np.ndarray
    reward_vector: np.ndarray
    next_obs: np.ndarray
    next_context: np.ndarray
    next_mask: np.ndarray
    next_seq_obs: np.ndarray
    done: float


class SumTree:
    def __init__(self, capacity: int):
        self.capacity = capacity
        self.tree = np.zeros(2 * capacity - 1, dtype=np.float32)
        self.data: List[Transition | None] = [None] * capacity
        self.write = 0
        self.size = 0

    def add(self, priority: float, data: Transition) -> None:
        index = self.write + self.capacity - 1
        self.data[self.write] = data
        self.update(index, priority)
        self.write = (self.write + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)

    def update(self, index: int, priority: float) -> None:
        delta = priority - self.tree[index]
        self.tree[index] = priority
        while index != 0:
            index = (index - 1) // 2
            self.tree[index] += delta

    def get(self, value: float) -> Tuple[int, float, Transition]:
        index = 0
        while index < self.capacity - 1:
            left = 2 * index + 1
            right = left + 1
            if value <= self.tree[left]:
                index = left
            else:
                value -= self.tree[left]
                index = right
        data_index = index - self.capacity + 1
        data = self.data[data_index]
        if data is None:
            raise RuntimeError("Sampled empty transition from replay buffer.")
        return index, self.tree[index], data

    @property
    def total(self) -> float:
        return float(self.tree[0])


class PrioritizedReplayBuffer:
    def __init__(
        self,
        capacity: int,
        alpha: float = 0.6,
        beta_start: float = 0.4,
        beta_increment: float = 1e-3,
        epsilon: float = 1e-5,
    ):
        self.alpha = alpha
        self.beta = beta_start
        self.beta_increment = beta_increment
        self.epsilon = epsilon
        self.tree = SumTree(capacity)
        self.max_priority = 1.0

    def add(self, transition: Transition, td_error: float | None = None) -> None:
        priority = self.max_priority if td_error is None else self._priority(td_error)
        self.tree.add(priority, transition)
        self.max_priority = max(self.max_priority, priority)

    def sample(self, batch_size: int) -> Tuple[List[Transition], np.ndarray, np.ndarray]:
        batch: List[Transition] = []
        indices: List[int] = []
        priorities: List[float] = []
        segment = self.tree.total / max(batch_size, 1)
        self.beta = min(1.0, self.beta + self.beta_increment)

        for batch_index in range(batch_size):
            start = segment * batch_index
            end = segment * (batch_index + 1)
            value = float(np.random.uniform(start, end))
            index, priority, data = self.tree.get(value)
            batch.append(data)
            indices.append(index)
            priorities.append(priority)

        probabilities = np.array(priorities, dtype=np.float32) / max(self.tree.total, 1e-8)
        weights = np.power(max(self.tree.size, 1) * probabilities, -self.beta)
        weights /= np.max(weights) if np.max(weights) > 0 else 1.0
        return batch, np.array(indices, dtype=np.int32), weights.astype(np.float32)

    def update_priorities(self, indices: np.ndarray, td_errors: np.ndarray) -> None:
        for index, td_error in zip(indices, td_errors):
            priority = self._priority(float(td_error))
            self.tree.update(int(index), priority)
            self.max_priority = max(self.max_priority, priority)

    def __len__(self) -> int:
        return self.tree.size

    def _priority(self, td_error: float) -> float:
        return float((abs(td_error) + self.epsilon) ** self.alpha)
