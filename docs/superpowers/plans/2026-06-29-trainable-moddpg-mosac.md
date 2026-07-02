# Trainable MO-DDPG and MO-SAC Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Upgrade MO-DDPG and MO-SAC from smoke policies to trainable actor-critic baselines while keeping the shared route-resource interface.

**Architecture:** Add a compact TensorFlow actor-critic base class in `src.models.baseline_algorithms`. MO-DDPG uses deterministic continuous control with route logits; MO-SAC uses stochastic exploration, twin critics, and an entropy term. `run_baseline_experiment(..., train_core=True)` runs short smoke training before evaluation.

**Tech Stack:** Python, NumPy, TensorFlow/Keras, existing simulator, existing `ActionRecord`, `unittest`.

---

### Task 1: Trainable Adapter API

**Files:**
- Modify: `tests/test_baseline_algorithms.py`
- Modify: `src/models/baseline_algorithms.py`

- [ ] **Step 1: Write failing tests**

Require `MoDdpgAdapter` and `MoSacAdapter` to expose `observe_transition`, `train_step`, `training_steps`, and `critic_count`; after collecting transitions, `train_step()` must update `training_steps` and return actor/critic losses.

- [ ] **Step 2: Run tests**

Run `python -m unittest tests.test_baseline_algorithms`. Expected: fail because smoke adapters do not train.

- [ ] **Step 3: Implement minimal actor-critic training**

Add TensorFlow actors, critics, target networks, replay storage, deterministic DDPG update, stochastic SAC-style update, and hard/soft target updates.

- [ ] **Step 4: Re-run tests**

Run `python -m unittest tests.test_baseline_algorithms`. Expected: pass.

### Task 2: Baseline Runner Training Hook

**Files:**
- Modify: `tests/test_baselines.py`
- Modify: `src/experiments/baseline_experiment.py`

- [ ] **Step 1: Write failing tests**

Require `run_baseline_experiment(..., include_core=True, train_core=True)` to include a `training_steps` column and nonzero training steps for MO-DDPG/MO-SAC.

- [ ] **Step 2: Run tests**

Run `python -m unittest tests.test_baselines`. Expected: fail because the runner has no training hook.

- [ ] **Step 3: Implement runner hook**

Before evaluating trainable core baselines, collect a short trajectory, call `observe_transition`, run `train_step`, and write `training_steps` to JSON/CSV.

- [ ] **Step 4: Re-run tests**

Run `python -m unittest tests.test_baselines`. Expected: pass.

### Task 3: Verification and Commit

**Files:**
- Modify: `README.md`

- [ ] **Step 1: Document scope**

Document that MO-DDPG/MO-SAC now have trainable actor-critic smoke implementations and manuscript-scale runs still require larger budgets.

- [ ] **Step 2: Verify**

Run `python -W error::ResourceWarning -m unittest discover -s tests` and `python reproduce.py --config configs/default.yaml`.

- [ ] **Step 3: Commit**

Commit on `feature/actor-critic-baselines` with message `feat: add trainable actor critic baselines`.
