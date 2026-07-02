from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

from src.evaluation.reporting import MANUSCRIPT_CSV_SCHEMAS
from src.models.baseline_catalog import baseline_catalog_entries
from src.utils.config import load_config, output_dirs
from src.utils.io import save_json, write_csv


REVIEW_PACKAGE_STATUS_FIELDS = [
    "execution_status",
    "generated_by",
    "requires_long_run",
    "notes",
]

PROTECTED_EXECUTED_RESULT_DIRS = {
    Path("outputs/results"),
    Path("outputs/figures"),
    Path("outputs/tables"),
}


@dataclass(frozen=True)
class ReviewManifestEntry:
    artifact: str
    paper_binding: str
    config: str
    script: str
    output_path: str
    status: str
    notes: str


def build_review_package(config_path: str | Path) -> dict[str, Any]:
    config = load_config(config_path)
    review = _section(config, "review_package")
    if not _as_bool(review.get("artifact_only", False)):
        raise ValueError("review_package.artifact_only must be true for the review package")

    dirs = output_dirs(config)
    overwrite = _as_bool(review.get("overwrite_executed_outputs", False))
    _assert_isolated_outputs(dirs, overwrite=overwrite)

    execution_status = str(review.get("execution_status", "not_executed"))
    generated_by = str(review.get("generated_by", "review_artifact_package"))
    requires_long_run = "true" if _as_bool(review.get("requires_long_run", True)) else "false"

    result_dir = dirs["result_dir"]
    created_csvs = _write_schema_csvs(
        result_dir,
        execution_status=execution_status,
        generated_by=generated_by,
        requires_long_run=requires_long_run,
    )
    manifest_path = _write_manifest(
        result_dir,
        config_path=Path(config_path),
        execution_status=execution_status,
    )
    catalog_path = _write_baseline_catalog(result_dir)
    audit_path = _write_audit_report(
        result_dir,
        config_path=Path(config_path),
        execution_status=execution_status,
        created_csvs=created_csvs,
    )
    summary = {
        "execution_status": execution_status,
        "artifact_only": True,
        "result_dir": str(result_dir),
        "figure_dir": str(dirs["figure_dir"]),
        "table_dir": str(dirs["table_dir"]),
        "created_csvs": [str(path) for path in created_csvs],
        "manifest": str(manifest_path),
        "baseline_catalog": str(catalog_path),
        "audit_report": str(audit_path),
    }
    save_json(result_dir / "review_package_summary.json", summary)
    return summary


def _write_schema_csvs(
    result_dir: Path,
    *,
    execution_status: str,
    generated_by: str,
    requires_long_run: str,
) -> list[Path]:
    created: list[Path] = []
    for filename, fields in MANUSCRIPT_CSV_SCHEMAS.items():
        row = {field: "" for field in fields}
        row.update(
            {
                "execution_status": execution_status,
                "generated_by": generated_by,
                "requires_long_run": requires_long_run,
                "notes": "Schema-complete review artifact package; no training or evaluation was executed.",
            }
        )
        path = write_csv(result_dir / filename, [row], fields + REVIEW_PACKAGE_STATUS_FIELDS)
        created.append(path)
    return created


def _write_manifest(result_dir: Path, *, config_path: Path, execution_status: str) -> Path:
    entries = [
        ReviewManifestEntry(
            artifact="overall_tradeoff.csv",
            paper_binding="Main trade-off table and core comparison figures",
            config=str(config_path),
            script="review_package.py",
            output_path=str(result_dir / "overall_tradeoff.csv"),
            status=execution_status,
            notes="Schema artifact; replace with executed results after long runs.",
        ),
        ReviewManifestEntry(
            artifact="generalization_seen_unseen.csv",
            paper_binding="Seen/unseen generalization diagnostics",
            config=str(config_path),
            script="review_package.py",
            output_path=str(result_dir / "generalization_seen_unseen.csv"),
            status=execution_status,
            notes="Schema artifact; generated without simulator evaluation.",
        ),
        ReviewManifestEntry(
            artifact="combined_ablation.csv",
            paper_binding="Ablation table and mechanism diagnostics",
            config=str(config_path),
            script="review_package.py",
            output_path=str(result_dir / "combined_ablation.csv"),
            status=execution_status,
            notes="Schema artifact; ablation runner is not invoked.",
        ),
        ReviewManifestEntry(
            artifact="statistical_tests.csv",
            paper_binding="Bootstrap and paired statistical reporting",
            config=str(config_path),
            script="review_package.py",
            output_path=str(result_dir / "statistical_tests.csv"),
            status=execution_status,
            notes="Schema artifact; statistical tests require executed seed-level rows.",
        ),
    ]
    return write_csv(
        result_dir / "experiment_manifest.csv",
        [entry.__dict__ for entry in entries],
        ["artifact", "paper_binding", "config", "script", "output_path", "status", "notes"],
    )


def _write_baseline_catalog(result_dir: Path) -> Path:
    rows = [
        {
            "method": entry.method,
            "layer": entry.layer,
            "role": entry.role,
            "implementation_status": entry.implementation_status,
            "reference_key": entry.reference_key,
            "adapter_module": entry.adapter_module,
            "execution_status": "not_executed",
            "notes": entry.notes,
        }
        for entry in baseline_catalog_entries()
    ]
    return write_csv(
        result_dir / "baseline_implementation_catalog.csv",
        rows,
        [
            "method",
            "layer",
            "role",
            "implementation_status",
            "reference_key",
            "adapter_module",
            "execution_status",
            "notes",
        ],
    )


def _write_audit_report(
    result_dir: Path,
    *,
    config_path: Path,
    execution_status: str,
    created_csvs: list[Path],
) -> Path:
    lines = [
        "# Review Artifact Package Audit Report",
        "",
        f"- Config: `{config_path}`",
        f"- Execution status: `{execution_status}`",
        "- Training executed: `false`",
        "- Evaluation executed: `false`",
        "- Numerical reproduction claimed: `false`",
        "- Purpose: reviewer-facing experiment artifact package.",
        "",
        "## Artifact Status",
        "",
    ]
    for path in created_csvs:
        lines.append(f"- `{path.name}`: schema-complete artifact, no numeric result values generated.")
    lines.extend(
        [
            "",
            "## Baseline Boundary",
            "",
            "The catalog reports local, structural-adaptation, public-code-adaptation, federated-adaptation, and paper-adaptation statuses. Schema artifacts do not imply that long-running baseline training has been executed.",
            "",
            "## Next Execution Stage",
            "",
            "Use smoke or paper-scale workflows only after replacing schema artifacts with executed result artifacts under a separate run directory.",
        ]
    )
    path = result_dir / "review_audit_report.md"
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


def _assert_isolated_outputs(dirs: Mapping[str, Path], *, overwrite: bool) -> None:
    if overwrite:
        return
    selected = {dirs["result_dir"], dirs["figure_dir"], dirs["table_dir"]}
    normalized = {Path(path.as_posix().rstrip("/")) for path in selected}
    if normalized & PROTECTED_EXECUTED_RESULT_DIRS:
        raise ValueError("review package outputs must not target normal executed-result directories")


def _section(mapping: Mapping[str, Any], key: str) -> Mapping[str, Any]:
    value = mapping.get(key, {})
    return value if isinstance(value, Mapping) else {}


def _as_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() in {"1", "true", "yes", "on"}
