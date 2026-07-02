from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.experiments.review_package import build_review_package


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate reviewer-facing artifact package without running experiments."
    )
    parser.add_argument("--config", type=Path, default=Path("configs/review_package.yaml"))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    summary = build_review_package(args.config)
    print(summary["result_dir"])


if __name__ == "__main__":
    main()
