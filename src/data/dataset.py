from __future__ import annotations

from dataclasses import dataclass

from src.data.schemas import Observation


@dataclass(frozen=True)
class GeneratedScenarioSample:
    seed: int
    mode: str
    observation: Observation

