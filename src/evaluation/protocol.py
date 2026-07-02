from __future__ import annotations

import math
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from src.evaluation.metrics import NormalizationBounds
from src.utils.io import save_json

OBJECTIVE_NAMES = ["latency", "energy", "violation", "privacy_gap"]


def scenario_seeds(paper_config: Any, *, split: str = "test", limit: int | None = None) -> list[int]:
    train_count, val_count, test_count = paper_config.protocol.train_val_test_split
    split_offsets = {
        "train": (0, train_count),
        "val": (train_count, val_count),
        "test": (train_count + val_count, test_count),
    }
    if split not in split_offsets:
        raise ValueError(f"Unknown split: {split}")
    offset, count = split_offsets[split]
    selected_count = min(count, limit) if limit is not None else count
    start = paper_config.protocol.split_seed + offset
    return [start + index for index in range(selected_count)]


def build_shared_normalization_bounds(
    records_by_method: Mapping[str, Sequence[dict[str, Any]]]
) -> NormalizationBounds:
    vectors = []
    for records in records_by_method.values():
        for record in records:
            vectors.append(_cost_vector(record))
    if not vectors:
        zeros = np.zeros(len(OBJECTIVE_NAMES), dtype=np.float32)
        ones = np.ones(len(OBJECTIVE_NAMES), dtype=np.float32)
        return NormalizationBounds(minimum=zeros, maximum=ones)
    stacked = np.stack(vectors).astype(np.float32)
    return NormalizationBounds(
        minimum=np.min(stacked, axis=0),
        maximum=np.max(stacked, axis=0),
    )


def save_normalization_bounds(path: str | Path, bounds: NormalizationBounds) -> Path:
    return save_json(
        path,
        {
            "objectives": OBJECTIVE_NAMES,
            "minimum": bounds.minimum.tolist(),
            "maximum": bounds.maximum.tolist(),
        },
    )


def load_normalization_bounds(path: str | Path) -> NormalizationBounds:
    import json

    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    return NormalizationBounds(
        minimum=np.asarray(payload["minimum"], dtype=np.float32),
        maximum=np.asarray(payload["maximum"], dtype=np.float32),
    )


def aggregate_metric_rows(
    rows: Sequence[Mapping[str, Any]],
    *,
    group_key: str,
    metrics: Sequence[str],
) -> list[dict[str, Any]]:
    grouped: dict[str, list[Mapping[str, Any]]] = {}
    for row in rows:
        grouped.setdefault(str(row[group_key]), []).append(row)

    summaries = []
    for group, group_rows in grouped.items():
        summary: dict[str, Any] = {group_key: group, "n": len(group_rows)}
        for metric in metrics:
            values = np.asarray([float(row[metric]) for row in group_rows], dtype=np.float64)
            mean = float(np.mean(values))
            std = float(np.std(values, ddof=1)) if values.size > 1 else 0.0
            ci95 = float(1.96 * std / math.sqrt(values.size)) if values.size > 1 else 0.0
            summary[f"{metric}_mean"] = mean
            summary[f"{metric}_std"] = std
            summary[f"{metric}_ci95"] = ci95
        summaries.append(summary)
    return summaries


def _cost_vector(record: Mapping[str, Any]) -> np.ndarray:
    value = record.get("cost_vector")
    if value is not None:
        return np.asarray(value, dtype=np.float32)
    return np.array(
        [
            float(record.get("latency", 0.0)),
            float(record.get("energy", 0.0)),
            1.0 if bool(record.get("violation", False)) else 0.0,
            0.0 if bool(record.get("privacy_match", False)) else 1.0,
        ],
        dtype=np.float32,
    )
