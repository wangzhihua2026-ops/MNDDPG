#!/usr/bin/env bash
set -euo pipefail

python train.py --config configs/proposed.yaml --rounds 1
