#!/usr/bin/env bash
set -euo pipefail

python - <<'PY'
from src.experiments.sensitivity_experiment import run_sensitivity_experiment

print(run_sensitivity_experiment("configs/sensitivity.yaml", rounds=1, steps=24))
PY
