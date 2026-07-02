# Experiment Closure Hardening Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make the MNDDPG experiment workflow more auditable by removing hard-coded evaluation budgets and marking placeholder workflow outputs honestly.

**Architecture:** Extend the existing YAML-to-dataclass config path, then consume those protocol fields in the federated runner and reproduction workflow. Keep MNDDPG model internals untouched.

**Tech Stack:** Python 3.10, dataclasses, unittest, TensorFlow-backed existing runner.

---

### Task 1: Parse Evaluation Protocol Fields

**Files:**
- Modify: `src/utils/paper_config.py`
- Modify: `src/utils/config.py`
- Test: `tests/test_config_protocol.py`

- [ ] **Step 1: Write failing tests**

Add tests asserting `evaluation.steps`, `evaluation.scenario_limit`, and `evaluation.bootstrap_resamples` are parsed into `PaperTrainingConfig.protocol`.

- [ ] **Step 2: Verify failure**

Run: `python tests/test_config_protocol.py`

Expected: FAIL because the protocol does not expose the new parsed fields yet.

- [ ] **Step 3: Implement minimal config parsing**

Add fields to `ExperimentProtocol`, parse them in `build_paper_config()`, and preserve explicit override precedence.

- [ ] **Step 4: Verify pass**

Run: `python tests/test_config_protocol.py`

Expected: PASS.

### Task 2: Use Configured Round Evaluation Steps

**Files:**
- Modify: `src/training/federated.py`
- Test: `tests/test_training.py`

- [ ] **Step 1: Write failing test**

Add a runner test with `evaluation_steps=3` and assert each training round calls evaluation with `3` steps.

- [ ] **Step 2: Verify failure**

Run: `python tests/test_training.py`

Expected: FAIL because `run_rounds()` uses `num_steps=24`.

- [ ] **Step 3: Implement minimal runner change**

Replace the hard-coded `24` with `self.config.protocol.evaluation_steps`.

- [ ] **Step 4: Verify pass**

Run: `python tests/test_training.py`

Expected: PASS.

### Task 3: Mark Placeholder Workflow Artifacts

**Files:**
- Modify: `src/experiments/reproduction_workflow.py`
- Test: `tests/test_reproduction_workflow.py`

- [ ] **Step 1: Write failing test**

Patch heavy workflow calls and assert the returned ablation/sensitivity metadata includes `status` values of `plan_only` and `grid_only`.

- [ ] **Step 2: Verify failure**

Run: `python tests/test_reproduction_workflow.py`

Expected: FAIL because current return values are plain strings.

- [ ] **Step 3: Implement minimal metadata return**

Return dictionaries for `ablation` and `sensitivity` with `path` and `status`.

- [ ] **Step 4: Verify pass**

Run: `python tests/test_reproduction_workflow.py`

Expected: PASS.

### Task 4: Focused Verification

**Files:**
- No production files unless failures reveal a scoped issue.

- [ ] **Step 1: Run focused tests**

Run: `python tests/test_config_protocol.py`, `python tests/test_training.py`, and `python tests/test_reproduction_workflow.py`.

Expected: PASS.

- [ ] **Step 2: Record known environment limitation**

Note in the final response that `pytest` is not installed in the active Python environment and full test discovery previously timed out due integration-style TensorFlow tests.
