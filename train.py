from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.experiments.main_experiment import run_main_experiment


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train the MNDDPG experiment workflow.")
    parser.add_argument("--config", type=Path, default=Path("configs/proposed.yaml"))
    parser.add_argument("--rounds", type=int, default=None)
    parser.add_argument("--output", type=Path, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    summary = run_main_experiment(args.config, rounds=args.rounds, output=args.output)
    print(summary["checkpoint"])


if __name__ == "__main__":
    main()

