from __future__ import annotations

from dataclasses import asdict
from pathlib import Path
from typing import Any

from src.training.checkpoint import save_weights_npz
from src.training.federated import PaperAlignedFederatedRunner
from src.experiments.tracking import write_experiment_run_context
from src.utils.config import build_paper_config, load_config, output_dirs
from src.utils.io import save_json, write_csv
from src.utils.logger import setup_logger
from src.utils.seed import set_global_seed


def run_main_experiment(
    config_path: str | Path,
    *,
    rounds: int | None = None,
    output: str | Path | None = None,
) -> dict[str, Any]:
    config_mapping = load_config(config_path)
    dirs = output_dirs(config_mapping)
    project = config_mapping.get("project", {})
    method = config_mapping.get("method", {})
    seed = int(project.get("seed", 3407)) if isinstance(project, dict) else 3407
    method_name = str(method.get("name", "MNDDPG")) if isinstance(method, dict) else "MNDDPG"

    set_global_seed(seed)
    logger = setup_logger("train", dirs["log_dir"])
    run_context = write_experiment_run_context(config_path, config_mapping, dirs)
    run_dir = Path(run_context["run_dir"])
    run_result_dir = run_dir / "results"
    run_checkpoint_dir = run_dir / "checkpoints"
    run_result_dir.mkdir(parents=True, exist_ok=True)
    run_checkpoint_dir.mkdir(parents=True, exist_ok=True)
    paper_config = build_paper_config(config_mapping, rounds=rounds)
    runner = PaperAlignedFederatedRunner(paper_config)
    summaries = runner.run_rounds(rounds=paper_config.federated.rounds)

    run_checkpoint_path = save_weights_npz(
        run_checkpoint_dir / "global_weights.npz",
        runner.global_agent.get_federated_weights(),
    )
    run_best_checkpoint_path = save_weights_npz(
        run_checkpoint_dir / "global_weights_best.npz",
        runner.get_best_weights(),
    )
    checkpoint_path = save_weights_npz(
        dirs["checkpoint_dir"] / "global_weights.npz",
        runner.global_agent.get_federated_weights(),
    )
    latest_checkpoint_path = save_weights_npz(
        dirs["checkpoint_dir"] / "global_weights_latest.npz",
        runner.global_agent.get_federated_weights(),
    )
    best_checkpoint_path = save_weights_npz(
        dirs["checkpoint_dir"] / "global_weights_best.npz",
        runner.get_best_weights(),
    )
    run_summary_path = run_result_dir / "train_summary.json"
    run_hv_path = run_result_dir / "hv_convergence.csv"
    run_comm_path = run_result_dir / "communication_efficiency.csv"
    summary_payload = {
        "method": method_name,
        "run_id": run_context["run_id"],
        "run_dir": run_context["run_dir"],
        "rounds": [asdict(summary) for summary in summaries],
        "checkpoint": str(checkpoint_path),
        "latest_checkpoint": str(latest_checkpoint_path),
        "best_checkpoint": str(best_checkpoint_path),
        "run_artifacts": {
            "result_dir": str(run_result_dir),
            "checkpoint_dir": str(run_checkpoint_dir),
            "train_summary": str(run_summary_path),
            "checkpoint": str(run_checkpoint_path),
            "best_checkpoint": str(run_best_checkpoint_path),
            "hv_convergence": str(run_hv_path),
            "communication_efficiency": str(run_comm_path),
        },
        "metadata": run_context,
    }
    result_path = Path(output) if output is not None else dirs["result_dir"] / "train_summary.json"
    save_json(run_summary_path, summary_payload)
    save_json(result_path, summary_payload)
    _write_training_csvs(run_result_dir, summaries, method_name)
    _write_training_csvs(dirs["result_dir"], summaries, method_name)
    logger.info("Saved training summary to %s", result_path)
    return summary_payload


def _write_training_csvs(result_dir: Path, summaries: list[Any], method: str) -> None:
    cumulative = 0
    hv_rows = []
    communication_rows = []
    for summary in summaries:
        cumulative += int(summary.communication_bytes)
        hv_rows.append(
            {
                "episode": summary.round_index,
                "round": summary.round_index,
                "method": method,
                "HV": summary.normalized_hv,
                "mean_latency": summary.mean_latency,
                "mean_energy": summary.mean_energy,
                "violation_rate": summary.violation_rate,
                "privacy_match": summary.privacy_match,
            }
        )
        communication_rows.append(
            {
                "round": summary.round_index,
                "method": method,
                "HV": summary.normalized_hv,
                "upload_bytes": summary.communication_bytes // 2,
                "download_bytes": summary.communication_bytes // 2,
                "cumulative_communication": cumulative,
            }
        )
    write_csv(
        result_dir / "hv_convergence.csv",
        hv_rows,
        [
            "episode",
            "round",
            "method",
            "HV",
            "mean_latency",
            "mean_energy",
            "violation_rate",
            "privacy_match",
        ],
    )
    write_csv(
        result_dir / "communication_efficiency.csv",
        communication_rows,
        ["round", "method", "HV", "upload_bytes", "download_bytes", "cumulative_communication"],
    )
