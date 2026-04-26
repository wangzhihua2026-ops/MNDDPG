from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

CURRENT_DIR = Path(__file__).resolve().parent
if str(CURRENT_DIR) not in sys.path:
    sys.path.insert(0, str(CURRENT_DIR))

from config import paper_reference_experiment
from runner import PublicExperimentRunner


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the public MNDDPG reference demo.")
    parser.add_argument("--rounds", type=int, default=3, help="Federated rounds to simulate.")
    parser.add_argument("--steps", type=int, default=24, help="Tasks per client per round.")
    parser.add_argument(
        "--output",
        type=Path,
        default=CURRENT_DIR / "outputs" / "demo_summary.json",
        help="Where to save the JSON summary.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = paper_reference_experiment()
    config = config.__class__(
        environment=config.environment,
        policy=config.policy,
        federated=config.federated,
        split_seed=config.split_seed,
        train_client_seeds=config.train_client_seeds,
        eval_scenario_seeds=config.eval_scenario_seeds,
        rounds=args.rounds,
        steps_per_round=args.steps,
    )
    runner = PublicExperimentRunner(config)
    summary = runner.run()

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(summary, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    print(json.dumps(summary["headline"], indent=2, ensure_ascii=False))
    print(f"\nSaved full summary to: {args.output}")


if __name__ == "__main__":
    main()

