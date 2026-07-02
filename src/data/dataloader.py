from __future__ import annotations

from collections.abc import Iterator

from src.data.environment import PaperAlignedEdgeOffloadingEnv
from src.data.schemas import Observation
from src.utils.paper_config import EnvironmentConfig


def iter_observations(
    config: EnvironmentConfig,
    *,
    seed: int,
    mode: str,
    steps: int,
) -> Iterator[Observation]:
    env = PaperAlignedEdgeOffloadingEnv(config, seed=seed, mode=mode)
    observation = env.reset()
    for _ in range(steps):
        yield observation
        observation = env._build_observation()

