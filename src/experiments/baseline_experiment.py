from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from src.data.environment import PaperAlignedEdgeOffloadingEnv
from src.evaluation.protocol import (
    aggregate_metric_rows,
    build_shared_normalization_bounds,
    save_normalization_bounds,
    scenario_seeds,
)
from src.evaluation.records import summarize_evaluation_records
from src.models.baseline import available_baseline_policies
from src.models.baseline_algorithms import available_core_baseline_adapters
from src.models.baseline_catalog import baseline_status_by_method
from src.utils.config import build_paper_config, load_config, output_dirs
from src.utils.io import save_json, write_csv


def run_baseline_experiment(
    config_path: str | Path,
    *,
    steps: int = 24,
    include_core: bool = False,
    train_core: bool = False,
    training_steps: int = 48,
    split: str = "test",
    scenario_limit: int | None = None,
) -> list[dict[str, Any]]:
    config_mapping = load_config(config_path)
    dirs = output_dirs(config_mapping)
    paper_config = build_paper_config(config_mapping)
    results = []
    statuses = baseline_status_by_method()
    policies = available_baseline_policies(seed=2026)
    if include_core:
        policies = {**policies, **available_core_baseline_adapters(seed=2026)}

    evaluation_seeds = scenario_seeds(paper_config, split=split, limit=scenario_limit)
    records_by_method_seed: dict[tuple[str, int], list[dict[str, Any]]] = {}
    records_by_method: dict[str, list[dict[str, Any]]] = {method: [] for method in policies}

    for method_name, policy in policies.items():
        if train_core:
            _train_policy_if_supported(policy, paper_config, steps=training_steps)

        for seed in evaluation_seeds:
            records = _evaluate_policy_records(policy, paper_config, seed=seed, steps=steps)
            records_by_method_seed[(method_name, seed)] = records
            records_by_method[method_name].extend(records)

    shared_bounds = build_shared_normalization_bounds(records_by_method)
    save_normalization_bounds(dirs["result_dir"] / "normalization_bounds.json", shared_bounds)
    seed_rows = []
    for (method_name, seed), records in records_by_method_seed.items():
        summary = summarize_evaluation_records(records, hv_seed=seed, bounds=shared_bounds)
        seed_rows.append(
            {
                "method": method_name,
                "scenario_seed": seed,
                "steps": steps,
                "latency": summary["mean_latency"],
                "energy": summary["mean_energy"],
                "violation_rate": summary["violation_rate"],
                "privacy_match": summary["privacy_match"],
                "reliability": 1.0 - float(summary["violation_rate"]),
                "normalized_hv": summary["normalized_hv"],
                "implementation_status": statuses.get(method_name, "uncataloged"),
                "training_steps": int(getattr(policies[method_name], "training_steps", 0)),
            }
        )

    for method_name in policies:
        method_rows = [row for row in seed_rows if row["method"] == method_name]
        results.append(
            {
                "method": method_name,
                "steps": steps,
                "mean_latency": float(np.mean([row["latency"] for row in method_rows])),
                "mean_energy": float(np.mean([row["energy"] for row in method_rows])),
                "violation_rate": float(np.mean([row["violation_rate"] for row in method_rows])),
                "privacy_match": float(np.mean([row["privacy_match"] for row in method_rows])),
                "reliability": float(np.mean([row["reliability"] for row in method_rows])),
                "normalized_hv": float(np.mean([row["normalized_hv"] for row in method_rows])),
                "implementation_status": statuses.get(method_name, "uncataloged"),
                "training_steps": int(getattr(policies[method_name], "training_steps", 0)),
            }
        )

    summary_rows = aggregate_metric_rows(
        seed_rows,
        group_key="method",
        metrics=("normalized_hv", "latency", "energy", "reliability", "privacy_match", "violation_rate"),
    )
    save_json(dirs["result_dir"] / "baseline_results.json", results)
    fieldnames = ["method", "steps", "mean_latency", "mean_energy", "violation_rate", "privacy_match"]
    if include_core or train_core:
        fieldnames.extend(["reliability", "normalized_hv", "implementation_status"])
    if train_core:
        fieldnames.append("training_steps")
    write_csv(
        dirs["result_dir"] / "baseline_results.csv",
        results,
        fieldnames,
    )
    write_csv(
        dirs["result_dir"] / "baseline_seed_results.csv",
        seed_rows,
        [
            "method",
            "scenario_seed",
            "steps",
            "latency",
            "energy",
            "violation_rate",
            "privacy_match",
            "reliability",
            "normalized_hv",
            "implementation_status",
            "training_steps",
        ],
    )
    write_csv(
        dirs["result_dir"] / "baseline_summary.csv",
        summary_rows,
        _summary_fieldnames(
            ("normalized_hv", "latency", "energy", "reliability", "privacy_match", "violation_rate")
        ),
    )
    _write_paper_tradeoff_csv(dirs["result_dir"] / "baseline_overall_tradeoff.csv", results)
    return results


def _train_policy_if_supported(policy: object, paper_config: object, *, steps: int) -> None:
    observe_transition = getattr(policy, "observe_transition", None)
    train_step = getattr(policy, "train_step", None)
    if observe_transition is None or train_step is None:
        return

    if hasattr(policy, "batch_size"):
        policy.batch_size = min(int(policy.batch_size), max(2, steps // 2))

    env = PaperAlignedEdgeOffloadingEnv(paper_config.environment, seed=3031, mode="seen")
    observation = env.reset()
    for _ in range(steps):
        action = policy.select_action(observation)
        next_observation, info = env.step(action)
        observe_transition(
            observation,
            action,
            info["reward_vector"],
            next_observation,
            False,
        )
        train_step()
        observation = next_observation


def _evaluate_policy_records(
    policy: object,
    paper_config: object,
    *,
    seed: int,
    steps: int,
) -> list[dict[str, Any]]:
    env = PaperAlignedEdgeOffloadingEnv(paper_config.environment, seed=seed, mode="seen")
    observation = env.reset()
    records = []
    for _ in range(steps):
        action = policy.select_action(observation)
        observation, info = env.step(action)
        records.append(info)
    return records


def _summary_fieldnames(metrics: tuple[str, ...]) -> list[str]:
    fields = ["method", "n"]
    for metric in metrics:
        fields.extend([f"{metric}_mean", f"{metric}_std", f"{metric}_ci95"])
    return fields


def _write_paper_tradeoff_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    write_csv(
        path,
        [
            {
                "seed": 2026,
                "scenario": "seen",
                "method": row["method"],
                "implementation_status": row["implementation_status"],
                "latency": row["mean_latency"],
                "energy": row["mean_energy"],
                "reliability": row["reliability"],
                "normalized_hv": row["normalized_hv"],
                "privacy_match": row["privacy_match"],
                "violation_rate": row["violation_rate"],
                "training_steps": row["training_steps"],
            }
            for row in rows
        ],
        [
            "seed",
            "scenario",
            "method",
            "implementation_status",
            "latency",
            "energy",
            "reliability",
            "normalized_hv",
            "privacy_match",
            "violation_rate",
            "training_steps",
        ],
    )
