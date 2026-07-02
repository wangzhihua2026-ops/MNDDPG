# Experiment Closure Hardening Design

## Goal

Upgrade the current MNDDPG experiment project from a smoke-test workflow to an auditable experiment loop without changing the core model architecture.

## Scope

This pass changes only the experiment-control layer:

- YAML configuration should drive evaluation steps, scenario limits, bootstrap counts, and run-output behavior.
- Training should stop using hard-coded evaluation budgets.
- Reproduction should clearly distinguish real executed results from schema or plan files.
- Existing latest-style output paths may remain for convenience, but run metadata should identify the exact produced artifacts.

This pass does not implement full CAPQL, PGMORL, FL-MO-SAC, or FedMORL fidelity. Those remain a later baseline-fidelity phase.

## Design

The project already has good module boundaries. `src.utils.config` maps YAML into `PaperTrainingConfig`; `src.training.federated` owns round execution; `src.experiments.reproduction_workflow` orchestrates smoke and paper workflows. The hardening work extends those existing boundaries instead of introducing a new experiment framework.

`ExperimentProtocol` will carry evaluation-facing controls such as `evaluation_steps`, `scenario_limit`, and `bootstrap_resamples`. `build_paper_config()` will parse the `evaluation:` section and preserve CLI overrides. `PaperAlignedFederatedRunner.run_rounds()` will use the configured evaluation budget instead of fixed `24` steps.

The reproduction workflow will return explicit artifact status for ablation and sensitivity outputs. When it only writes a plan or grid, the returned metadata will say so. When later workflows run the real experiments, they can use the same keys with `status="executed"`.

## Testing

Tests will be added before production edits:

- A config test proves `configs/paper_main.yaml` maps `evaluation.steps`, `scenario_limit`, and `bootstrap_resamples` into the protocol object.
- A federated runner test proves round evaluation uses the configured step count instead of a hard-coded value.
- A reproduction workflow test proves ablation and sensitivity entries are reported as plan/grid artifacts rather than executed results when the workflow does not run them.

The existing environment lacks `pytest`, so focused verification will use `python -m unittest` for the affected test modules.
