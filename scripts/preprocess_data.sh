#!/usr/bin/env bash
set -euo pipefail

python - <<'PY'
from src.data.preprocess import preprocess_generated_data

path = preprocess_generated_data("data/raw", "data/processed")
print(path)
PY

