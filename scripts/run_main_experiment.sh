#!/usr/bin/env bash
set -euo pipefail

python train.py --config configs/proposed.yaml --rounds 1
python evaluate.py --config configs/proposed.yaml --checkpoint outputs/checkpoints/global_weights.npz --mode seen --steps 24

