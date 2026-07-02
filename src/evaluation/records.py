from __future__ import annotations

from typing import Any, Sequence

import numpy as np

from src.evaluation.metrics import (
    calibration_bounds,
    monte_carlo_hypervolume,
    normalized_utilities,
    NormalizationBounds,
)
from src.models.modules import EXPERT_NAMES


def summarize_evaluation_records(
    records: Sequence[dict[str, Any]],
    *,
    hv_seed: int = 0,
    hv_samples: int = 2048,
    bounds: NormalizationBounds | None = None,
) -> dict[str, Any]:
    if not records:
        return {
            "num_records": 0,
            "mean_latency": 0.0,
            "mean_energy": 0.0,
            "violation_rate": 0.0,
            "privacy_match": 0.0,
            "normalized_hv": 0.0,
            "expert_entropy": 0.0,
            "privacy_sensitivity_rows": [],
            "expert_activation_rows": [],
            "violation_diagnostic_rows": [],
        }

    costs = np.stack([_cost_vector(record) for record in records]).astype(np.float32)
    effective_bounds = bounds or calibration_bounds(costs)
    utilities = np.clip(normalized_utilities(costs, effective_bounds), 0.0, 1.0)
    normalized_hv = monte_carlo_hypervolume(utilities, num_samples=hv_samples, seed=hv_seed)

    violation_values = np.array([bool(record.get("violation", False)) for record in records])
    privacy_values = np.array([bool(record.get("privacy_match", False)) for record in records])
    expert_indices = np.array([int(record.get("expert_index", 0)) for record in records])
    expert_counts = np.bincount(expert_indices, minlength=len(EXPERT_NAMES)).astype(np.float64)
    expert_probs = expert_counts / max(float(np.sum(expert_counts)), 1.0)
    expert_entropy = float(-np.sum(expert_probs * np.log(expert_probs + 1e-12)))

    return {
        "num_records": len(records),
        "mean_latency": float(np.mean([float(record["latency"]) for record in records])),
        "mean_energy": float(np.mean([float(record["energy"]) for record in records])),
        "violation_rate": float(np.mean(violation_values)),
        "privacy_match": float(np.mean(privacy_values)),
        "normalized_hv": float(normalized_hv),
        "expert_entropy": expert_entropy,
        "privacy_sensitivity_rows": _privacy_sensitivity_rows(records),
        "expert_activation_rows": _expert_activation_rows(expert_probs, expert_entropy),
        "violation_diagnostic_rows": [
            {
                "violation_rate": float(np.mean(violation_values)),
            }
        ],
    }


def _cost_vector(record: dict[str, Any]) -> np.ndarray:
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


def _privacy_sensitivity_rows(records: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
    rows = []
    for group_name, predicate in (
        ("low", lambda level: level < 5),
        ("medium", lambda level: 5 <= level < 8),
        ("high", lambda level: level >= 8),
    ):
        group = [
            record
            for record in records
            if predicate(int(record.get("privacy_level", 0)))
        ]
        if group:
            privacy_match = float(np.mean([bool(row.get("privacy_match", False)) for row in group]))
            high_violations = [
                bool(row.get("violation", False))
                for row in group
                if int(row.get("privacy_level", 0)) >= 8
            ]
            high_violation_rate = (
                float(np.mean(high_violations)) if high_violations else 0.0
            )
        else:
            privacy_match = 0.0
            high_violation_rate = 0.0
        rows.append(
            {
                "privacy_group": group_name,
                "privacy_policy_match": privacy_match,
                "high_sensitivity_violation_rate": high_violation_rate,
            }
        )
    return rows


def _expert_activation_rows(
    expert_probs: np.ndarray, expert_entropy: float
) -> list[dict[str, Any]]:
    return [
        {
            "expert": expert,
            "activation_probability": float(expert_probs[index]),
            "expert_entropy": expert_entropy,
        }
        for index, expert in enumerate(EXPERT_NAMES)
    ]
