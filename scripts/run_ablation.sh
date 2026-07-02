#!/usr/bin/env bash
set -euo pipefail

python - <<'PY'
from src.experiments.ablation_experiment import run_ablation_experiment

print(run_ablation_experiment("configs/ablation.yaml", rounds=1, steps=24))
PY
