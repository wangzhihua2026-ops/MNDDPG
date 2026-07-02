from __future__ import annotations

import csv
from pathlib import Path
from typing import Any

from src.utils.config import load_config, output_dirs
from src.utils.visualization import (
    write_markdown_table,
    write_svg_bar_chart,
    write_svg_line_chart,
)


def generate_paper_artifacts(config_path: str | Path) -> dict[str, str]:
    dirs = output_dirs(load_config(config_path))
    result_dir = dirs["result_dir"]
    figure_dir = dirs["figure_dir"]
    table_dir = dirs["table_dir"]
    artifacts: dict[str, str] = {}

    baseline_rows = _read_csv(result_dir / "baseline_results.csv")
    if baseline_rows:
        baseline_table = write_markdown_table(
            table_dir / "baseline_results.md",
            ["method", "mean_latency", "mean_energy", "violation_rate", "privacy_match"],
            [
                [
                    row.get("method", ""),
                    _fmt(row.get("mean_latency", "")),
                    _fmt(row.get("mean_energy", "")),
                    _fmt(row.get("violation_rate", "")),
                    _fmt(row.get("privacy_match", "")),
                ]
                for row in baseline_rows
            ],
        )
        artifacts["baseline_table"] = str(baseline_table)
        artifacts["baseline_latency"] = str(
            write_svg_bar_chart(
                figure_dir / "baseline_latency.svg",
                title="Baseline mean latency",
                labels=[row.get("method", "") for row in baseline_rows],
                values=[_float(row.get("mean_latency", 0.0)) for row in baseline_rows],
                y_label="Mean latency",
            )
        )
        artifacts["baseline_energy"] = str(
            write_svg_bar_chart(
                figure_dir / "baseline_energy.svg",
                title="Baseline mean energy",
                labels=[row.get("method", "") for row in baseline_rows],
                values=[_float(row.get("mean_energy", 0.0)) for row in baseline_rows],
                y_label="Mean energy",
            )
        )

    baseline_tradeoff_rows = _read_csv(result_dir / "baseline_overall_tradeoff.csv")
    if baseline_tradeoff_rows:
        baseline_tradeoff_table = write_markdown_table(
            table_dir / "baseline_overall_tradeoff.md",
            [
                "method",
                "status",
                "latency",
                "energy",
                "reliability",
                "normalized_hv",
                "privacy_match",
            ],
            [
                [
                    row.get("method", ""),
                    row.get("implementation_status", ""),
                    _fmt(row.get("latency", "")),
                    _fmt(row.get("energy", "")),
                    _fmt(row.get("reliability", "")),
                    _fmt(row.get("normalized_hv", "")),
                    _fmt(row.get("privacy_match", "")),
                ]
                for row in baseline_tradeoff_rows
            ],
        )
        artifacts["baseline_tradeoff_table"] = str(baseline_tradeoff_table)
        artifacts["baseline_hv"] = str(
            write_svg_bar_chart(
                figure_dir / "baseline_hv.svg",
                title="Baseline normalized HV",
                labels=[row.get("method", "") for row in baseline_tradeoff_rows],
                values=[_float(row.get("normalized_hv", 0.0)) for row in baseline_tradeoff_rows],
                y_label="Normalized HV",
            )
        )

    hv_rows = _read_csv(result_dir / "hv_convergence.csv")
    if hv_rows:
        artifacts["hv_convergence"] = str(
            write_svg_line_chart(
                figure_dir / "hv_convergence.svg",
                title="HV convergence",
                x_values=[_float(row.get("round", index + 1)) for index, row in enumerate(hv_rows)],
                y_values=[_float(row.get("HV", 0.0)) for row in hv_rows],
                x_label="Round",
                y_label="HV",
            )
        )

    tradeoff_rows = _read_csv(result_dir / "overall_tradeoff.csv")
    if tradeoff_rows:
        tradeoff_table = write_markdown_table(
            table_dir / "overall_tradeoff.md",
            ["scenario", "method", "latency", "energy", "reliability", "normalized_hv"],
            [
                [
                    row.get("scenario", ""),
                    row.get("method", ""),
                    _fmt(row.get("latency", "")),
                    _fmt(row.get("energy", "")),
                    _fmt(row.get("reliability", "")),
                    _fmt(row.get("normalized_hv", "")),
                ]
                for row in tradeoff_rows
            ],
        )
        artifacts["overall_tradeoff_table"] = str(tradeoff_table)
        artifacts["overall_tradeoff_hv"] = str(
            write_svg_bar_chart(
                figure_dir / "overall_tradeoff_hv.svg",
                title="Overall normalized HV",
                labels=[f"{row.get('method', '')}-{row.get('scenario', '')}" for row in tradeoff_rows],
                values=[_float(row.get("normalized_hv", 0.0)) for row in tradeoff_rows],
                y_label="Normalized HV",
            )
        )

    statistical_rows = _read_csv(result_dir / "statistical_tests.csv")
    if statistical_rows:
        statistical_table = write_markdown_table(
            table_dir / "statistical_tests.md",
            [
                "reference",
                "paired",
                "metric",
                "mean_difference",
                "bootstrap_ci",
                "p_value",
                "n_pairs",
            ],
            [
                [
                    row.get("reference_method", ""),
                    row.get("paired_method", ""),
                    row.get("metric", ""),
                    _fmt(row.get("mean_difference", "")),
                    row.get("bootstrap_ci", ""),
                    _fmt(row.get("p_value", "")),
                    row.get("n_pairs", ""),
                ]
                for row in statistical_rows
            ],
        )
        artifacts["statistical_tests_table"] = str(statistical_table)

    return artifacts


def _read_csv(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _float(value: Any) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return 0.0


def _fmt(value: Any) -> str:
    try:
        return f"{float(value):.4f}"
    except (TypeError, ValueError):
        return str(value)
