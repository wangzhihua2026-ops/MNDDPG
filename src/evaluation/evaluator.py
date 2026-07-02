from __future__ import annotations

from pathlib import Path
from typing import Any

from src.evaluation.protocol import (
    aggregate_metric_rows,
    build_shared_normalization_bounds,
    save_normalization_bounds,
    scenario_seeds,
)
from src.training.checkpoint import load_weights_npz
from src.training.federated import PaperAlignedFederatedRunner
from src.evaluation.records import summarize_evaluation_records
from src.utils.io import save_json, write_csv
from src.utils.paper_config import PaperTrainingConfig


def evaluate_runner(
    config: PaperTrainingConfig,
    *,
    mode: str,
    steps: int,
    checkpoint: str | Path | None = None,
    output_path: str | Path | None = None,
) -> dict[str, Any]:
    runner = PaperAlignedFederatedRunner(config)
    if checkpoint is not None and Path(checkpoint).exists():
        runner.global_agent.set_federated_weights(load_weights_npz(checkpoint))

    stats = runner.evaluate(mode=mode, num_steps=steps)
    stats["mode"] = mode
    stats["steps"] = steps
    if output_path is not None:
        save_json(output_path, _json_safe_summary(stats))
    return stats


def evaluate_runner_scenarios(
    config: PaperTrainingConfig,
    *,
    mode: str,
    steps: int,
    split: str = "test",
    scenario_limit: int | None = None,
    checkpoint: str | Path | None = None,
    output_dir: str | Path | None = None,
) -> dict[str, Any]:
    runner = PaperAlignedFederatedRunner(config)
    if checkpoint is not None and Path(checkpoint).exists():
        runner.global_agent.set_federated_weights(load_weights_npz(checkpoint))

    seeds = scenario_seeds(config, split=split, limit=scenario_limit)
    records_by_seed: dict[int, list[dict[str, Any]]] = {}
    for seed in seeds:
        records_by_seed[seed] = _evaluate_runner_records(
            runner,
            mode=mode,
            seed=seed,
            steps=steps,
        )

    bounds = build_shared_normalization_bounds(
        {str(seed): records for seed, records in records_by_seed.items()}
    )
    seed_rows = []
    for seed, records in records_by_seed.items():
        summary = summarize_evaluation_records(records, hv_seed=seed, bounds=bounds)
        seed_rows.append(
            {
                "method": "MNDDPG",
                "scenario_seed": seed,
                "mode": mode,
                "steps": steps,
                "latency": summary["mean_latency"],
                "energy": summary["mean_energy"],
                "violation_rate": summary["violation_rate"],
                "privacy_match": summary["privacy_match"],
                "reliability": 1.0 - float(summary["violation_rate"]),
                "normalized_hv": summary["normalized_hv"],
            }
        )
    summary_rows = aggregate_metric_rows(
        seed_rows,
        group_key="method",
        metrics=("normalized_hv", "latency", "energy", "reliability", "privacy_match", "violation_rate"),
    )
    if output_dir is not None:
        directory = Path(output_dir)
        directory.mkdir(parents=True, exist_ok=True)
        save_normalization_bounds(directory / "normalization_bounds.json", bounds)
        write_csv(
            directory / "evaluation_seed_results.csv",
            seed_rows,
            [
                "method",
                "scenario_seed",
                "mode",
                "steps",
                "latency",
                "energy",
                "violation_rate",
                "privacy_match",
                "reliability",
                "normalized_hv",
            ],
        )
        write_csv(
            directory / "evaluation_summary.csv",
            summary_rows,
            _summary_fieldnames(
                ("normalized_hv", "latency", "energy", "reliability", "privacy_match", "violation_rate")
            ),
        )
    return {"seed_rows": seed_rows, "summary_rows": summary_rows}


def overall_tradeoff_row(stats: dict[str, Any], *, seed: int, method: str) -> dict[str, Any]:
    return {
        "seed": seed,
        "scenario": stats.get("mode", "seen"),
        "method": method,
        "latency": stats.get("mean_latency", ""),
        "energy": stats.get("mean_energy", ""),
        "reliability": 1.0 - float(stats.get("violation_rate", 0.0)),
        "normalized_hv": stats.get("normalized_hv", 0.0),
    }


def _evaluate_runner_records(
    runner: PaperAlignedFederatedRunner,
    *,
    mode: str,
    seed: int,
    steps: int,
) -> list[dict[str, Any]]:
    from src.data.environment import PaperAlignedEdgeOffloadingEnv

    env = PaperAlignedEdgeOffloadingEnv(runner.config.environment, seed=seed, mode=mode)
    runner.global_agent.reset_episode()
    observation = env.reset()
    records = []
    for _ in range(steps):
        action = runner.global_agent.select_action(observation, training=False)
        next_observation, info = env.step(action)
        runner.global_agent.commit_observation(next_observation.vector)
        records.append(info)
        observation = next_observation
    return records


def _summary_fieldnames(metrics: tuple[str, ...]) -> list[str]:
    fields = ["method", "n"]
    for metric in metrics:
        fields.extend([f"{metric}_mean", f"{metric}_std", f"{metric}_ci95"])
    return fields


def privacy_sensitivity_rows(
    stats: dict[str, Any], *, seed: int, method: str
) -> list[dict[str, Any]]:
    return [
        {
            "seed": seed,
            "method": method,
            **row,
        }
        for row in stats.get("privacy_sensitivity_rows", [])
    ]


def expert_activation_rows(
    stats: dict[str, Any], *, seed: int, method: str
) -> list[dict[str, Any]]:
    regime = stats.get("mode", "seen")
    return [
        {
            "seed": seed,
            "regime": regime,
            "method": method,
            **row,
        }
        for row in stats.get("expert_activation_rows", [])
    ]


def violation_diagnostic_rows(
    stats: dict[str, Any], *, seed: int, method: str
) -> list[dict[str, Any]]:
    return [
        {
            "seed": seed,
            "method": method,
            **row,
        }
        for row in stats.get("violation_diagnostic_rows", [])
    ]


def _json_safe_summary(stats: dict[str, Any]) -> dict[str, Any]:
    return {
        key: value
        for key, value in stats.items()
        if key != "records"
    }
