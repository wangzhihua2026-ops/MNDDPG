from __future__ import annotations

import shutil
from pathlib import Path
from typing import Any, Mapping, Sequence

from src.evaluation.evaluator import (
    evaluate_runner_scenarios,
    evaluate_runner,
    expert_activation_rows,
    overall_tradeoff_row,
    privacy_sensitivity_rows,
    violation_diagnostic_rows,
)
from src.evaluation.reporting import create_manuscript_schema_files
from src.evaluation.statistical_test import write_statistical_tests
from src.experiments.ablation_experiment import run_ablation_experiment, write_ablation_plan
from src.experiments.artifact_generation import generate_paper_artifacts
from src.experiments.baseline_experiment import run_baseline_experiment
from src.experiments.main_experiment import run_main_experiment
from src.experiments.sensitivity_experiment import (
    run_sensitivity_experiment,
    write_sensitivity_grid,
)
from src.models.baseline_catalog import write_baseline_catalog_files
from src.utils.config import build_paper_config, load_config, output_dirs
from src.utils.io import write_csv


def run_reproduction_workflow(
    config_path: str | Path,
    *,
    rounds: int | None = None,
    steps: int = 24,
    scenario_limit: int = 2,
    baseline_training_steps: int | None = None,
    bootstrap_resamples: int = 256,
    execute_ablation: bool = True,
    execute_sensitivity: bool = True,
    ablation_groups: Sequence[str] | None = None,
    sensitivity_grid: Mapping[str, Sequence[float]] | None = None,
) -> dict[str, Any]:
    schema_files = create_manuscript_schema_files(config_path)
    train_summary = run_main_experiment(config_path, rounds=rounds)
    config_mapping = load_config(config_path)
    dirs = output_dirs(config_mapping)
    paper_config = build_paper_config(config_mapping)
    checkpoint = train_summary["checkpoint"]

    seen = evaluate_runner(
        paper_config,
        mode="seen",
        steps=steps,
        checkpoint=checkpoint,
        output_path=dirs["result_dir"] / "evaluation_seen.json",
    )
    unseen = evaluate_runner(
        paper_config,
        mode="unseen",
        steps=steps,
        checkpoint=checkpoint,
        output_path=dirs["result_dir"] / "evaluation_unseen.json",
    )
    baseline = run_baseline_experiment(
        config_path,
        steps=steps,
        include_core=True,
        train_core=True,
        training_steps=baseline_training_steps or max(8, steps * 2),
        scenario_limit=scenario_limit,
    )
    baseline_catalog = write_baseline_catalog_files(config_path)
    _write_main_result_rows(dirs["result_dir"], seen, unseen)
    scenario_stats = evaluate_runner_scenarios(
        paper_config,
        mode="seen",
        steps=steps,
        scenario_limit=scenario_limit,
        checkpoint=checkpoint,
        output_dir=dirs["result_dir"],
    )
    statistical_path = _write_statistical_tests_from_seed_rows(
        dirs["result_dir"],
        bootstrap_resamples=bootstrap_resamples,
    )
    if execute_ablation:
        ablation_rows = run_ablation_experiment(
            config_path,
            groups=ablation_groups,
            rounds=rounds,
            steps=steps,
        )
        ablation_output = {
            "path": str(dirs["result_dir"] / "combined_ablation.csv"),
            "status": "executed",
            "row_count": len(ablation_rows),
            "description": "Ablation groups were trained and evaluated by this workflow.",
        }
    else:
        ablation_path = write_ablation_plan(config_path)
        ablation_output = {
            "path": str(ablation_path),
            "status": "plan_only",
            "row_count": 0,
            "description": "Schema and ablation-group plan written; ablation training was not executed by this workflow.",
        }

    if execute_sensitivity:
        sensitivity_rows = run_sensitivity_experiment(
            config_path,
            parameter_grid=sensitivity_grid,
            rounds=rounds,
            steps=steps,
        )
        sensitivity_output = {
            "path": str(dirs["result_dir"] / "sensitivity_results.csv"),
            "status": "executed",
            "row_count": len(sensitivity_rows),
            "description": "Sensitivity trials were trained and evaluated by this workflow.",
        }
    else:
        sensitivity_path = write_sensitivity_grid(config_path)
        sensitivity_output = {
            "path": str(sensitivity_path),
            "status": "grid_only",
            "row_count": 0,
            "description": "Sensitivity grid written; sensitivity trials were not executed by this workflow.",
        }
    artifacts = generate_paper_artifacts(config_path)
    archived_artifacts = _archive_workflow_artifacts(
        dirs["result_dir"],
        train_summary.get("run_dir"),
    )

    return {
        "train_summary": train_summary,
        "run_artifacts": train_summary.get("run_artifacts", {}),
        "seen": seen,
        "unseen": unseen,
        "baseline": baseline,
        "baseline_catalog": {key: str(path) for key, path in baseline_catalog.items()},
        "schema_files": [str(path) for path in schema_files],
        "scenario_evaluation": scenario_stats,
        "statistical_tests": str(statistical_path),
        "ablation": ablation_output,
        "sensitivity": sensitivity_output,
        "artifacts": artifacts,
        "archived_artifacts": archived_artifacts,
    }


def _write_main_result_rows(result_dir: Path, seen: dict[str, Any], unseen: dict[str, Any]) -> None:
    overall_rows = [
        overall_tradeoff_row(seen, seed=2026, method="MNDDPG"),
        overall_tradeoff_row(unseen, seed=2026, method="MNDDPG"),
    ]
    write_csv(
        result_dir / "overall_tradeoff.csv",
        overall_rows,
        ["seed", "scenario", "method", "latency", "energy", "reliability", "normalized_hv"],
    )

    privacy_rows = privacy_sensitivity_rows(seen, seed=2026, method="MNDDPG")
    write_csv(
        result_dir / "privacy_sensitivity.csv",
        privacy_rows,
        [
            "seed",
            "method",
            "privacy_group",
            "privacy_policy_match",
            "high_sensitivity_violation_rate",
        ],
    )

    expert_rows = expert_activation_rows(seen, seed=2026, method="MNDDPG")
    write_csv(
        result_dir / "expert_activation.csv",
        expert_rows,
        [
            "seed",
            "regime",
            "method",
            "expert",
            "activation_probability",
            "expert_entropy",
        ],
    )

    violation_rows = violation_diagnostic_rows(seen, seed=2026, method="MNDDPG")
    write_csv(
        result_dir / "violation_diagnostics.csv",
        violation_rows,
        ["seed", "method", "violation_rate"],
    )

    seen_hv = float(seen.get("normalized_hv", 0.0))
    unseen_hv = float(unseen.get("normalized_hv", 0.0))
    write_csv(
        result_dir / "generalization_seen_unseen.csv",
        [
            {
                "seed": 2026,
                "method": "MNDDPG",
                "seen_unseen": "seen",
                "normalized_hv": seen_hv,
                "generalization_drop": 0.0,
            },
            {
                "seed": 2026,
                "method": "MNDDPG",
                "seen_unseen": "unseen",
                "normalized_hv": unseen_hv,
                "generalization_drop": seen_hv - unseen_hv,
            },
        ],
        ["seed", "method", "seen_unseen", "normalized_hv", "generalization_drop"],
    )


def _write_statistical_tests_from_seed_rows(
    result_dir: Path,
    *,
    bootstrap_resamples: int,
) -> Path:
    import csv

    evaluation_path = result_dir / "evaluation_seed_results.csv"
    baseline_path = result_dir / "baseline_seed_results.csv"
    merged_path = result_dir / "paired_seed_results.csv"
    output_path = result_dir / "statistical_tests.csv"

    rows = []
    for path in (evaluation_path, baseline_path):
        if not path.exists():
            continue
        with path.open(newline="", encoding="utf-8") as handle:
            rows.extend(list(csv.DictReader(handle)))

    write_csv(
        merged_path,
        rows,
        ["method", "scenario_seed", "normalized_hv", "latency"],
    )
    write_statistical_tests(
        merged_path,
        output_path,
        proposed_method="MNDDPG",
        metric="normalized_hv",
        resamples=bootstrap_resamples,
        seed=2026,
    )
    return output_path


def _archive_workflow_artifacts(
    result_dir: Path,
    run_dir_value: object,
) -> dict[str, Any]:
    if not run_dir_value:
        return {"run_dir": "", "files": []}

    run_dir = Path(str(run_dir_value))
    archive_dir = run_dir / "artifacts"
    archive_dir.mkdir(parents=True, exist_ok=True)
    artifact_names = [
        "evaluation_seen.json",
        "evaluation_unseen.json",
        "overall_tradeoff.csv",
        "privacy_sensitivity.csv",
        "expert_activation.csv",
        "violation_diagnostics.csv",
        "generalization_seen_unseen.csv",
        "evaluation_seed_results.csv",
        "baseline_seed_results.csv",
        "paired_seed_results.csv",
        "statistical_tests.csv",
        "combined_ablation.csv",
        "sensitivity_results.csv",
    ]
    archived_files: list[str] = []
    for name in artifact_names:
        source = result_dir / name
        if source.exists():
            destination = archive_dir / name
            shutil.copy2(source, destination)
            archived_files.append(str(destination))
    return {"run_dir": str(run_dir), "files": archived_files}
