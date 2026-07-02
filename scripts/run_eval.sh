#!/usr/bin/env bash
set -euo pipefail

python evaluate.py --config configs/proposed.yaml --checkpoint outputs/checkpoints/global_weights_best.npz --mode seen --steps 24
