from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from typing import Any, Mapping, Sequence

from src.training.federated import PaperAlignedFederatedRunner
from src.utils.config import build_paper_config
from src.utils.config import load_config, output_dirs
from src.utils.io import write_csv


DEFAULT_PARAMETER_GRID: dict[str, list[float]] = {
    "clip_norm": [1.0, 1.75, 2.5],
    "gating_temperature": [0.8, 0.45, 0.2],
}


def run_sensitivity_experiment(
    config_path: str | Path,
    *,
    parameter_grid: Mapping[str, Sequence[float]] | None = None,
    rounds: int | None = None,
    steps: int = 24,
) -> list[dict[str, Any]]:
    config_mapping = load_config(config_path)
    dirs = output_dirs(config_mapping)
    grid = dict(parameter_grid) if parameter_grid is not None else _grid_from_config(config_mapping)
    rows: list[dict[str, Any]] = []

    for parameter, values in grid.items():
        for value in values:
            trial_config = _sensitivity_config(config_mapping, parameter, value)
            paper_config = build_paper_config(trial_config, rounds=rounds)
            runner = PaperAlignedFederatedRunner(paper_config)
            runner.run_rounds(rounds=paper_config.federated.rounds)
            stats = runner.evaluate(mode="seen", num_steps=steps)
            rows.append(
                {
                    "parameter": parameter,
                    "value": value,
                    "normalized_hv": stats["normalized_hv"],
                    "privacy_match": stats["privacy_match"],
                    "violation_rate": stats["violation_rate"],
                }
            )

    return write_sensitivity_rows(dirs["result_dir"] / "sensitivity_results.csv", rows)


def write_sensitivity_grid(config_path: str | Path) -> Path:
    config_mapping = load_config(config_path)
    dirs = output_dirs(config_mapping)
    rows = []
    for clip_norm in (1.0, 1.75, 2.5):
        rows.append({"parameter": "clip_norm", "value": clip_norm, "normalized_hv": ""})
    for temperature in (0.8, 0.45, 0.2):
        rows.append({"parameter": "gumbel_temperature", "value": temperature, "normalized_hv": ""})
    return write_csv(
        dirs["result_dir"] / "sensitivity_results.csv",
        rows,
        ["parameter", "value", "normalized_hv"],
    )


def write_sensitivity_rows(path: str | Path, rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    write_csv(
        path,
        rows,
        ["parameter", "value", "normalized_hv", "privacy_match", "violation_rate"],
    )
    return rows


def _grid_from_config(config_mapping: Mapping[str, Any]) -> dict[str, list[float]]:
    sensitivity = config_mapping.get("sensitivity", {})
    if not isinstance(sensitivity, Mapping):
        return DEFAULT_PARAMETER_GRID
    grid: dict[str, list[float]] = {}
    for parameter, default_values in DEFAULT_PARAMETER_GRID.items():
        raw = sensitivity.get(f"{parameter}_values")
        grid[parameter] = _parse_values(raw, default_values)
    return grid


def _parse_values(raw: Any, default_values: Sequence[float]) -> list[float]:
    if raw is None:
        return list(default_values)
    if isinstance(raw, (list, tuple)):
        return [float(value) for value in raw]
    return [float(value.strip()) for value in str(raw).split(",") if value.strip()]


def _sensitivity_config(
    config_mapping: Mapping[str, Any], parameter: str, value: float
) -> dict[str, Any]:
    trial = deepcopy(dict(config_mapping))
    if parameter == "clip_norm":
        trial.setdefault("federated", {})["clip_norm"] = float(value)
    elif parameter in {"gating_temperature", "route_temperature"}:
        trial.setdefault("agent", {})[parameter] = float(value)
    else:
        raise ValueError(f"Unsupported sensitivity parameter: {parameter}")
    return trial
