# Result Artifact Generation Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Generate paper-facing Markdown tables and SVG figures from existing experiment CSV outputs, then include those artifacts in the reproduction workflow.

**Architecture:** Extend `src.utils.visualization` with dependency-free Markdown and SVG helpers. Add `src.experiments.artifact_generation` as the orchestration layer that reads CSV files from `outputs/results` and writes figures/tables to `outputs/figures` and `outputs/tables`. `run_reproduction_workflow` calls it after producing CSV outputs.

**Tech Stack:** Python standard library, SVG text output, existing config/output-dir utilities, `unittest`.

---

### Task 1: Visualization Helpers

**Files:**
- Modify: `src/utils/visualization.py`
- Create: `tests/test_visualization.py`

- [ ] **Step 1: Write failing tests**

Require `write_svg_bar_chart` and `write_svg_line_chart` to create valid SVG files containing labels and numeric marks.

- [ ] **Step 2: Run tests**

Run `python -m unittest tests.test_visualization`. Expected: fail because SVG helpers do not exist.

- [ ] **Step 3: Implement minimal helpers**

Add small SVG writers with stable dimensions, escaped labels, axes, bars/line path, and no external plotting dependency.

- [ ] **Step 4: Re-run tests**

Run `python -m unittest tests.test_visualization`. Expected: pass.

### Task 2: Artifact Orchestrator

**Files:**
- Create: `src/experiments/artifact_generation.py`
- Create: `tests/test_artifact_generation.py`

- [ ] **Step 1: Write failing tests**

Create sample CSV files and require `generate_paper_artifacts("configs/default.yaml")` to write `baseline_results.md`, `baseline_latency.svg`, and `hv_convergence.svg`.

- [ ] **Step 2: Run tests**

Run `python -m unittest tests.test_artifact_generation`. Expected: fail because module does not exist.

- [ ] **Step 3: Implement orchestrator**

Read `baseline_results.csv`, `overall_tradeoff.csv`, and `hv_convergence.csv` when present. Write Markdown summary tables and SVG figures. Skip missing CSV files gracefully.

- [ ] **Step 4: Re-run tests**

Run `python -m unittest tests.test_artifact_generation`. Expected: pass.

### Task 3: Reproduction Integration

**Files:**
- Modify: `src/experiments/reproduction_workflow.py`
- Modify: `tests/test_reporting.py`
- Modify: `README.md`

- [ ] **Step 1: Write failing test**

Require `run_reproduction_workflow` to return an `artifacts` key and create at least one SVG in `outputs/figures`.

- [ ] **Step 2: Run test**

Run the focused reporting test. Expected: fail before integration.

- [ ] **Step 3: Integrate and document**

Call `generate_paper_artifacts` at the end of the reproduction workflow and document generated figures/tables in README.

- [ ] **Step 4: Verify**

Run `python -W error::ResourceWarning -m unittest discover -s tests` and `python reproduce.py --config configs/default.yaml`.

- [ ] **Step 5: Commit**

Commit on `feature/result-artifacts` with message `feat: generate paper result artifacts`.
