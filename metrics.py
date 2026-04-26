from __future__ import annotations

from collections import Counter
from typing import Dict, Sequence

import numpy as np


def privacy_group(level: int, medium_threshold: int = 5, high_threshold: int = 8) -> str:
    if level >= high_threshold:
        return "high"
    if level >= medium_threshold:
        return "medium"
    return "low"


def normalize_utilities(cost_vectors: np.ndarray) -> np.ndarray:
    mins = np.min(cost_vectors, axis=0)
    maxs = np.max(cost_vectors, axis=0)
    return 1.0 - (cost_vectors - mins) / (maxs - mins + 1e-8)


def approximate_hypervolume(
    points: np.ndarray, num_samples: int = 12000, seed: int = 0
) -> float:
    if points.size == 0:
        return 0.0
    rng = np.random.default_rng(seed)
    samples = rng.uniform(0.0, 1.0, size=(num_samples, points.shape[1]))
    dominated = np.any(
        np.all(points[:, None, :] >= samples[None, :, :], axis=2),
        axis=0,
    )
    return float(np.mean(dominated))


def summarize_records(records: Sequence[Dict[str, object]]) -> Dict[str, object]:
    if not records:
        return {}

    cost_vectors = np.array([record["cost_vector"] for record in records], dtype=np.float64)
    reward_vectors = np.array(
        [record["reward_vector"] for record in records],
        dtype=np.float64,
    )
    scenario_costs = {}
    for record in records:
        seed = int(record.get("scenario_seed", -1))
        scenario_costs.setdefault(seed, []).append(record["cost_vector"])
    scenario_vectors = np.array(
        [np.mean(vectors, axis=0) for _, vectors in sorted(scenario_costs.items())],
        dtype=np.float64,
    )
    utilities = normalize_utilities(scenario_vectors)
    hv = approximate_hypervolume(utilities)
    privacy_counts = {"low": 0, "medium": 0, "high": 0}
    privacy_hits = {"low": 0, "medium": 0, "high": 0}
    expert_counter = Counter()

    for record in records:
        group = privacy_group(int(record["privacy_level"]))
        privacy_counts[group] += 1
        privacy_hits[group] += int(bool(record["privacy_match"]))
        expert_counter[str(record["expert_name"])] += 1

    privacy_match_by_group = {
        group: privacy_hits[group] / max(privacy_counts[group], 1)
        for group in privacy_counts
    }

    return {
        "num_records": len(records),
        "num_scenarios": int(len(scenario_vectors)),
        "mean_cost_vector": cost_vectors.mean(axis=0).tolist(),
        "mean_reward_vector": reward_vectors.mean(axis=0).tolist(),
        "hypervolume": hv,
        "avg_latency": float(np.mean([record["latency"] for record in records])),
        "avg_energy": float(np.mean([record["energy"] for record in records])),
        "violation_rate": float(np.mean([record["violation"] for record in records])),
        "privacy_match_by_group": privacy_match_by_group,
        "expert_usage": dict(expert_counter),
    }
