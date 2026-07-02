from __future__ import annotations

import numpy as np


def clip_unit_interval(values: np.ndarray) -> np.ndarray:
    return np.clip(values, 0.0, 1.0)

