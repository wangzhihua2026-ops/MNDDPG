from __future__ import annotations

import importlib.metadata
import platform
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping

from src.utils.config import build_paper_config
from src.utils.io import save_json, write_csv


def write_experiment_run_context(
    config_path: str | Path,
    config_mapping: Mapping[str, Any],
    dirs: Mapping[str, Path],
    *,
    run_id: str | None = None,
) -> dict[str, str]:
    """Write auditable metadata that ties a run to config, seeds, and environment."""

    resolved_run_id = run_id or _build_run_id(config_mapping)
    run_dir = Path(dirs["result_dir"]) / "runs" / resolved_run_id
    run_dir.mkdir(parents=True, exist_ok=True)

    paper_config = build_paper_config(config_mapping)
    save_json(
        run_dir / "config_snapshot.json",
        {
            "config_path": str(Path(config_path)),
            "config": dict(config_mapping),
        },
    )
    save_json(run_dir / "environment.json", _environment_payload())
    save_json(run_dir / "seed_split.json", _seed_split_payload(paper_config))
    write_csv(
        run_dir / "scenario_manifest.csv",
        _scenario_manifest_rows(paper_config),
        ["scenario_id", "split", "seed", "mode"],
    )

    return {
        "run_id": resolved_run_id,
        "run_dir": str(run_dir),
        "config_snapshot": str(run_dir / "config_snapshot.json"),
        "environment": str(run_dir / "environment.json"),
        "seed_split": str(run_dir / "seed_split.json"),
        "scenario_manifest": str(run_dir / "scenario_manifest.csv"),
    }


def _build_run_id(config_mapping: Mapping[str, Any]) -> str:
    project = config_mapping.get("project", {})
    method = config_mapping.get("method", {})
    seed = project.get("seed", 3407) if isinstance(project, Mapping) else 3407
    method_name = method.get("name", "MNDDPG") if isinstance(method, Mapping) else "MNDDPG"
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    clean_method = str(method_name).lower().replace(" ", "_").replace("-", "_")
    return f"{clean_method}_seed{seed}_{timestamp}"


def _environment_payload() -> dict[str, Any]:
    package_names = ["numpy", "tensorflow", "pandas", "matplotlib", "PyYAML", "pytest"]
    packages: dict[str, str] = {}
    for package in package_names:
        try:
            packages[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            packages[package] = "not-installed"
    return {
        "python": sys.version,
        "executable": sys.executable,
        "platform": platform.platform(),
        "packages": packages,
    }


def _seed_split_payload(paper_config: Any) -> dict[str, Any]:
    protocol = paper_config.protocol
    train_count, val_count, test_count = protocol.train_val_test_split
    return {
        "split_seed": protocol.split_seed,
        "train_count": train_count,
        "val_count": val_count,
        "test_count": test_count,
        "independent_run_seeds": list(protocol.independent_run_seeds),
    }


def _scenario_manifest_rows(paper_config: Any) -> list[dict[str, Any]]:
    protocol = paper_config.protocol
    train_count, val_count, test_count = protocol.train_val_test_split
    rows: list[dict[str, Any]] = []
    cursor = protocol.split_seed
    for split, count in (
        ("train", train_count),
        ("val", val_count),
        ("test", test_count),
    ):
        for index in range(count):
            rows.append(
                {
                    "scenario_id": f"{split}_{index:04d}",
                    "split": split,
                    "seed": cursor,
                    "mode": "seen",
                }
            )
            cursor += 1
    return rows
