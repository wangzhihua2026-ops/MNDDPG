from __future__ import annotations

from pathlib import Path

import numpy as np


_SEP = "__slash__"


def save_weights_npz(path: str | Path, weights: dict[str, np.ndarray]) -> Path:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    safe_weights = {key.replace("/", _SEP): value for key, value in weights.items()}
    np.savez_compressed(output_path, **safe_weights)
    return output_path


def load_weights_npz(path: str | Path) -> dict[str, np.ndarray]:
    checkpoint_path = Path(path)
    with np.load(checkpoint_path, allow_pickle=False) as data:
        return {key.replace(_SEP, "/"): data[key] for key in data.files}

