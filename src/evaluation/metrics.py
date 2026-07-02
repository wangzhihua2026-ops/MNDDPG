from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, List, Sequence

import numpy as np


@dataclass(frozen=True)
class NormalizationBounds:
    minimum: np.ndarray
    maximum: np.ndarray


def calibration_bounds(cost_vectors: Sequence[np.ndarray]) -> NormalizationBounds:
    stacked = np.asarray(cost_vectors, dtype=np.float32)
    return NormalizationBounds(
        minimum=np.min(stacked, axis=0),
        maximum=np.max(stacked, axis=0),
    )


def normalized_utilities(
    scenario_costs: Sequence[np.ndarray], bounds: NormalizationBounds
) -> np.ndarray:
    costs = np.asarray(scenario_costs, dtype=np.float32)
    return 1.0 - (costs - bounds.minimum) / (bounds.maximum - bounds.minimum + 1e-8)


def monte_carlo_hypervolume(
    utilities: np.ndarray, num_samples: int = 20000, seed: int = 0
) -> float:
    if utilities.size == 0:
        return 0.0
    rng = np.random.default_rng(seed)
    samples = rng.uniform(0.0, 1.0, size=(num_samples, utilities.shape[1]))
    dominated = np.any(
        np.all(utilities[:, None, :] >= samples[None, :, :], axis=2),
        axis=0,
    )
    return float(np.mean(dominated))
