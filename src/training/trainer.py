from __future__ import annotations

from src.training.federated import PaperAlignedFederatedRunner
from src.utils.paper_config import PaperTrainingConfig


def build_runner(config: PaperTrainingConfig) -> PaperAlignedFederatedRunner:
    return PaperAlignedFederatedRunner(config)

