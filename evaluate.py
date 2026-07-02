from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.evaluation.evaluator import (
    evaluate_runner,
    evaluate_runner_scenarios,
    overall_tradeoff_row,
)
from src.utils.config import build_paper_config, load_config, output_dirs
from src.utils.io import write_csv


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate an MNDDPG checkpoint.")
    parser.add_argument("--config", type=Path, default=Path("configs/proposed.yaml"))
    parser.add_argument("--checkpoint", type=Path, default=None)
    parser.add_argument("--mode", choices=("seen", "unseen"), default="seen")
    parser.add_argument("--steps", type=int, default=24)
    parser.add_argument("--split", choices=("train", "val", "test"), default="test")
    parser.add_argument("--scenario-limit", type=int, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config_mapping = load_config(args.config)
    dirs = output_dirs(config_mapping)
    paper_config = build_paper_config(config_mapping)
    if args.scenario_limit is not None:
        stats = evaluate_runner_scenarios(
            paper_config,
            mode=args.mode,
            steps=args.steps,
            split=args.split,
            scenario_limit=args.scenario_limit,
            checkpoint=args.checkpoint,
            output_dir=dirs["result_dir"],
        )
        print(dirs["result_dir"] / "evaluation_summary.csv")
        return
    output_path = dirs["result_dir"] / f"evaluation_{args.mode}.json"
    stats = evaluate_runner(
        paper_config,
        mode=args.mode,
        steps=args.steps,
        checkpoint=args.checkpoint,
        output_path=output_path,
    )
    write_csv(
        dirs["result_dir"] / "overall_tradeoff.csv",
        [overall_tradeoff_row(stats, seed=2026, method="MNDDPG")],
        ["seed", "scenario", "method", "latency", "energy", "reliability", "normalized_hv"],
    )
    print(output_path)


if __name__ == "__main__":
    main()
