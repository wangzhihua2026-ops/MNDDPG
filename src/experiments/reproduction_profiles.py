from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Mapping


@dataclass(frozen=True)
class ReproductionProfileSettings:
    profile: str
    config_path: Path
    rounds: int | None
    steps: int
    scenario_limit: int
    baseline_training_steps: int
    bootstrap_resamples: int
    ablation_groups: tuple[str, ...] | None
    sensitivity_grid: Mapping[str, tuple[float, ...]] | None


PROFILE_DEFAULTS: dict[str, ReproductionProfileSettings] = {
    "smoke": ReproductionProfileSettings(
        profile="smoke",
        config_path=Path("configs/default.yaml"),
        rounds=1,
        steps=24,
        scenario_limit=2,
        baseline_training_steps=48,
        bootstrap_resamples=256,
        ablation_groups=("Full MNDDPG", "w/o Privacy Mask"),
        sensitivity_grid={"clip_norm": (1.75,)},
    ),
    "paper": ReproductionProfileSettings(
        profile="paper",
        config_path=Path("configs/paper_main.yaml"),
        rounds=None,
        steps=200,
        scenario_limit=30,
        baseline_training_steps=400,
        bootstrap_resamples=2000,
        ablation_groups=None,
        sensitivity_grid=None,
    ),
}


def resolve_reproduction_profile(
    profile: str,
    *,
    config_path: Path | None = None,
    rounds: int | None = None,
    steps: int | None = None,
    scenario_limit: int | None = None,
    baseline_training_steps: int | None = None,
    bootstrap_resamples: int | None = None,
) -> ReproductionProfileSettings:
    if profile not in PROFILE_DEFAULTS:
        raise ValueError(f"Unknown reproduction profile: {profile}")
    base = PROFILE_DEFAULTS[profile]
    return ReproductionProfileSettings(
        profile=base.profile,
        config_path=config_path or base.config_path,
        rounds=base.rounds if rounds is None else rounds,
        steps=base.steps if steps is None else steps,
        scenario_limit=base.scenario_limit if scenario_limit is None else scenario_limit,
        baseline_training_steps=(
            base.baseline_training_steps
            if baseline_training_steps is None
            else baseline_training_steps
        ),
        bootstrap_resamples=(
            base.bootstrap_resamples if bootstrap_resamples is None else bootstrap_resamples
        ),
        ablation_groups=base.ablation_groups,
        sensitivity_grid=base.sensitivity_grid,
    )
