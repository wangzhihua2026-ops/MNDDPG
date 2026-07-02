from __future__ import annotations

import numpy as np


def proxy_scalar_reward(proxy_weights: np.ndarray, reward_vector: np.ndarray) -> float:
    return float(np.dot(proxy_weights, reward_vector))

