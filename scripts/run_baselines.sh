#!/usr/bin/env bash
set -euo pipefail

python - <<'PY'
from src.experiments.baseline_experiment import run_baseline_experiment

print(
    run_baseline_experiment(
        "configs/baseline.yaml",
        steps=24,
        include_core=True,
        train_core=True,
        training_steps=48,
    )
)
PY
