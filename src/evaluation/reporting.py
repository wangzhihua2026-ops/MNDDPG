from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path

from src.utils.config import load_config, output_dirs
from src.utils.io import write_csv


MANUSCRIPT_CSV_SCHEMAS: dict[str, list[str]] = {
    "overall_tradeoff.csv": [
        "seed",
        "scenario",
        "method",
        "latency",
        "energy",
        "reliability",
        "normalized_hv",
    ],
    "privacy_sensitivity.csv": [
        "seed",
        "method",
        "privacy_group",
        "privacy_policy_match",
        "high_sensitivity_violation_rate",
    ],
    "hv_convergence.csv": [
        "episode",
        "round",
        "method",
        "HV",
        "mean_latency",
        "mean_energy",
        "violation_rate",
        "privacy_match",
    ],
    "communication_efficiency.csv": [
        "round",
        "method",
        "HV",
        "upload_bytes",
        "download_bytes",
        "cumulative_communication",
    ],
    "generalization_seen_unseen.csv": [
        "seed",
        "method",
        "seen_unseen",
        "normalized_hv",
        "generalization_drop",
    ],
    "stress_load.csv": [
        "seed",
        "method",
        "load_level",
        "violation_rate",
        "latency",
        "reliability",
    ],
    "combined_ablation.csv": [
        "seed",
        "ablation_group",
        "HV",
        "privacy_match",
        "violation_rate",
    ],
    "expert_activation.csv": [
        "seed",
        "regime",
        "method",
        "expert",
        "activation_probability",
        "expert_entropy",
    ],
    "violation_diagnostics.csv": [
        "seed",
        "method",
        "violation_rate",
    ],
    "statistical_tests.csv": [
        "reference_method",
        "paired_method",
        "metric",
        "mean_difference",
        "bootstrap_ci",
        "p_value",
        "n_pairs",
    ],
}


@dataclass(frozen=True)
class CsvSchemaValidation:
    path: str
    valid: bool
    expected: list[str]
    actual: list[str]


def create_manuscript_schema_files(config_path: str | Path) -> list[Path]:
    config_mapping = load_config(config_path)
    dirs = output_dirs(config_mapping)
    created = []
    for filename, fieldnames in MANUSCRIPT_CSV_SCHEMAS.items():
        created.append(write_csv(dirs["result_dir"] / filename, [], fieldnames))
    return created


def validate_csv_schema(
    path: str | Path,
    expected_fieldnames: list[str],
) -> CsvSchemaValidation:
    csv_path = Path(path)
    if not csv_path.exists():
        return CsvSchemaValidation(
            path=str(csv_path),
            valid=False,
            expected=expected_fieldnames,
            actual=[],
        )
    with csv_path.open(newline="", encoding="utf-8") as handle:
        reader = csv.reader(handle)
        actual = next(reader, [])
    return CsvSchemaValidation(
        path=str(csv_path),
        valid=actual == expected_fieldnames,
        expected=expected_fieldnames,
        actual=actual,
    )


def validate_manuscript_result_schemas(config_path: str | Path) -> list[CsvSchemaValidation]:
    config_mapping = load_config(config_path)
    dirs = output_dirs(config_mapping)
    return [
        validate_csv_schema(dirs["result_dir"] / filename, fieldnames)
        for filename, fieldnames in MANUSCRIPT_CSV_SCHEMAS.items()
    ]
