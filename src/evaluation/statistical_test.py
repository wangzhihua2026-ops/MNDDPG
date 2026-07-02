from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import numpy as np

from src.utils.io import write_csv


@dataclass(frozen=True)
class BootstrapResult:
    mean_difference: float
    ci_low: float
    ci_high: float
    p_value: float


def paired_bootstrap(
    left: Sequence[float],
    right: Sequence[float],
    *,
    resamples: int = 1000,
    seed: int = 0,
) -> BootstrapResult:
    left_arr = np.asarray(left, dtype=np.float64)
    right_arr = np.asarray(right, dtype=np.float64)
    if left_arr.shape != right_arr.shape:
        raise ValueError("Paired bootstrap inputs must have the same shape.")
    if left_arr.size == 0:
        raise ValueError("Paired bootstrap requires at least one pair.")

    rng = np.random.default_rng(seed)
    differences = left_arr - right_arr
    sample_means = []
    for _ in range(resamples):
        indices = rng.integers(0, differences.size, size=differences.size)
        sample_means.append(float(np.mean(differences[indices])))

    distribution = np.asarray(sample_means, dtype=np.float64)
    mean_difference = float(np.mean(differences))
    ci_low, ci_high = np.percentile(distribution, [2.5, 97.5])
    p_value = float(2.0 * min(np.mean(distribution <= 0), np.mean(distribution >= 0)))
    return BootstrapResult(
        mean_difference=mean_difference,
        ci_low=float(ci_low),
        ci_high=float(ci_high),
        p_value=min(p_value, 1.0),
    )


STATISTICAL_TEST_FIELDS = [
    "reference_method",
    "paired_method",
    "metric",
    "mean_difference",
    "bootstrap_ci",
    "p_value",
    "n_pairs",
]


def write_statistical_tests(
    seed_results_path: str | Path,
    output_path: str | Path,
    *,
    proposed_method: str = "MNDDPG",
    metric: str = "normalized_hv",
    resamples: int = 1000,
    seed: int = 0,
) -> list[dict[str, object]]:
    rows = _read_seed_rows(seed_results_path)
    methods = sorted({row["method"] for row in rows if row["method"] != proposed_method})
    output_rows: list[dict[str, object]] = []
    for method in methods:
        paired = _paired_metric_values(rows, proposed_method, method, metric)
        if not paired:
            continue
        left, right = zip(*paired)
        result = paired_bootstrap(left, right, resamples=resamples, seed=seed)
        output_rows.append(
            {
                "reference_method": proposed_method,
                "paired_method": method,
                "metric": metric,
                "mean_difference": result.mean_difference,
                "bootstrap_ci": f"[{result.ci_low:.6f}, {result.ci_high:.6f}]",
                "p_value": result.p_value,
                "n_pairs": len(paired),
            }
        )
    write_csv(output_path, output_rows, STATISTICAL_TEST_FIELDS)
    return output_rows


def _read_seed_rows(path: str | Path) -> list[dict[str, str]]:
    with Path(path).open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _paired_metric_values(
    rows: Sequence[dict[str, str]],
    reference_method: str,
    paired_method: str,
    metric: str,
) -> list[tuple[float, float]]:
    by_method_seed = {
        (row["method"], row["scenario_seed"]): float(row[metric])
        for row in rows
        if row.get(metric, "") != ""
    }
    seeds = sorted(
        {
            row["scenario_seed"]
            for row in rows
            if row["method"] in {reference_method, paired_method}
        }
    )
    paired = []
    for scenario_seed in seeds:
        left_key = (reference_method, scenario_seed)
        right_key = (paired_method, scenario_seed)
        if left_key in by_method_seed and right_key in by_method_seed:
            paired.append((by_method_seed[left_key], by_method_seed[right_key]))
    return paired
