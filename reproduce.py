from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.experiments.reproduction_workflow import run_reproduction_workflow
from src.experiments.reproduction_profiles import resolve_reproduction_profile


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the MNDDPG reproduction workflow.")
    parser.add_argument("--profile", choices=("smoke", "paper"), default="smoke")
    parser.add_argument("--config", type=Path, default=None)
    parser.add_argument("--rounds", type=int, default=None)
    parser.add_argument("--steps", type=int, default=None)
    parser.add_argument("--scenario-limit", type=int, default=None)
    parser.add_argument("--baseline-training-steps", type=int, default=None)
    parser.add_argument("--bootstrap-resamples", type=int, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    settings = resolve_reproduction_profile(
        args.profile,
        config_path=args.config,
        rounds=args.rounds,
        steps=args.steps,
        scenario_limit=args.scenario_limit,
        baseline_training_steps=args.baseline_training_steps,
        bootstrap_resamples=args.bootstrap_resamples,
    )
    outputs = run_reproduction_workflow(
        settings.config_path,
        rounds=settings.rounds,
        steps=settings.steps,
        scenario_limit=settings.scenario_limit,
        baseline_training_steps=settings.baseline_training_steps,
        bootstrap_resamples=settings.bootstrap_resamples,
        ablation_groups=settings.ablation_groups,
        sensitivity_grid=settings.sensitivity_grid,
    )
    print(outputs["schema_files"][0])


if __name__ == "__main__":
    main()
