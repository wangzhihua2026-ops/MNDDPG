from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, Tuple

import numpy as np


@dataclass(frozen=True)
class ModelDelta:
    delta: Dict[str, np.ndarray]
    num_samples: int


def parameter_l2_norm(delta: Dict[str, np.ndarray]) -> float:
    total = 0.0
    for value in delta.values():
        total += float(np.sum(np.square(value)))
    return float(np.sqrt(total))


def clip_delta(delta: Dict[str, np.ndarray], clip_norm: float) -> Dict[str, np.ndarray]:
    norm = parameter_l2_norm(delta)
    if norm <= clip_norm or norm == 0.0:
        return {key: value.copy() for key, value in delta.items()}
    factor = clip_norm / norm
    return {key: value * factor for key, value in delta.items()}


def aggregate_deltas(
    base_params: Dict[str, np.ndarray],
    deltas: Iterable[ModelDelta],
    clip_norm: float,
) -> Tuple[Dict[str, np.ndarray], int]:
    deltas = list(deltas)
    if not deltas:
        return {key: value.copy() for key, value in base_params.items()}, 0

    clipped = [ModelDelta(clip_delta(item.delta, clip_norm), item.num_samples) for item in deltas]
    total_samples = sum(item.num_samples for item in clipped)
    averaged = {key: np.zeros_like(value) for key, value in base_params.items()}

    for item in clipped:
        weight = item.num_samples / max(total_samples, 1)
        for key, value in item.delta.items():
            averaged[key] += weight * value

    updated = {key: base_params[key] + averaged[key] for key in base_params}
    model_bytes = sum(value.nbytes for value in base_params.values())
    communication_bytes = model_bytes * (len(clipped) + len(clipped))
    return updated, communication_bytes

