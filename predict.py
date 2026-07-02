from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data.environment import PaperAlignedEdgeOffloadingEnv
from src.models.proposed_model import PaperAlignedMNDDPGAgent
from src.utils.config import build_paper_config, load_config


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run one MNDDPG action prediction.")
    parser.add_argument("--config", type=Path, default=Path("configs/proposed.yaml"))
    parser.add_argument("--seed", type=int, default=2026)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    paper_config = build_paper_config(load_config(args.config))
    env = PaperAlignedEdgeOffloadingEnv(paper_config.environment, seed=args.seed, mode="seen")
    observation = env.reset()
    agent = PaperAlignedMNDDPGAgent(paper_config.agent, seed=args.seed)
    action = agent.select_action(observation, training=False)
    print(
        {
            "route_index": action.route_index,
            "continuous_action": action.continuous_action.tolist(),
            "expert_index": action.expert_index,
        }
    )


if __name__ == "__main__":
    main()

