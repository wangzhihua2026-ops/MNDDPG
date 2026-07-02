# Review Artifact Package Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a reviewer-facing artifact package that exposes the paper experiment protocol, schema-complete outputs, manifest, and audit report without running experiments.

**Architecture:** Add a `review_package` layer around the existing experiment framework. The layer writes schema artifacts under `outputs/review_package/`, validates output isolation, and documents the workflow in README. Existing training, evaluation, baseline, ablation, sensitivity, and reporting modules remain the source of executable experiment logic.

**Tech Stack:** Python standard library, existing `src.utils.io.write_csv`, existing `src.evaluation.reporting.MANUSCRIPT_CSV_SCHEMAS`, YAML config files, `unittest`.

---

## Tasks

- [x] Add `configs/review_package.yaml` with `review_package.artifact_only: true`, `execution_status: not_executed`, and isolated output directories.
- [x] Add `src/experiments/review_package.py` to write schema-complete CSV artifacts, `experiment_manifest.csv`, `baseline_implementation_catalog.csv`, `review_audit_report.md`, and `review_package_summary.json`.
- [x] Add `review_package.py` as the top-level CLI entry.
- [x] Add `tests/test_review_package.py` to cover schema fields, output isolation, and no calls to training or evaluation workflows.
- [x] Update `README.md` with reviewer-facing package instructions.

## Verification Plan

When execution is allowed, run:

```bash
python -m unittest tests.test_review_package tests.test_config_protocol -q
python review_package.py --config configs/review_package.yaml
```

Expected behavior: tests pass, and the package command writes only under `outputs/review_package/`.
