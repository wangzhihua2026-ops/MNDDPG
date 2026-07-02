# Review Artifact Package Design

## Goal

Upgrade the MNDDPG experiment project into a reviewer-facing artifact package that exposes the full paper experiment protocol, artifact schema, command surface, and audit trail without running training or fabricating numerical results in this phase.

## Scope

The package provides a `review_package` profile, schema-complete result artifacts, an experiment manifest, a baseline implementation catalog, and an audit report. It keeps all output under `outputs/review_package/` so the review artifacts do not overwrite executed experiment outputs.

The phase does not run training, evaluation, dependency installation, or statistical testing. Numeric result fields remain empty unless produced by a later executed workflow.

## Artifact Contract

The package covers the manuscript-facing outputs:

- `overall_tradeoff.csv`
- `privacy_sensitivity.csv`
- `hv_convergence.csv`
- `communication_efficiency.csv`
- `generalization_seen_unseen.csv`
- `stress_load.csv`
- `combined_ablation.csv`
- `expert_activation.csv`
- `violation_diagnostics.csv`
- `statistical_tests.csv`
- `baseline_implementation_catalog.csv`
- `experiment_manifest.csv`
- `review_audit_report.md`

Each CSV includes status metadata such as `execution_status`, `generated_by`, `requires_long_run`, and `notes`.

## Acceptance Criteria

- The repository contains a review artifact profile and documented workflow.
- The artifact package exposes the full paper experiment output surface.
- The generated artifacts cannot be confused with executed numerical results.
- The manifest maps paper evidence to code, config, outputs, and current status.
- The audit report honestly states the execution boundary.
