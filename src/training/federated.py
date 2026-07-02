from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Sequence, Tuple

import numpy as np

from src.data.environment import PaperAlignedEdgeOffloadingEnv
from src.evaluation.records import summarize_evaluation_records
from src.models.proposed_model import PaperAlignedMNDDPGAgent
from src.utils.paper_config import PaperTrainingConfig, paper_training_config


def l2_norm(delta: Dict[str, np.ndarray]) -> float:
    return float(np.sqrt(sum(np.sum(np.square(value)) for value in delta.values())))


def clip_delta(delta: Dict[str, np.ndarray], clip_norm: float) -> Dict[str, np.ndarray]:
    norm = l2_norm(delta)
    if norm <= clip_norm or norm == 0.0:
        return {key: value.copy() for key, value in delta.items()}
    factor = clip_norm / norm
    return {key: value * factor for key, value in delta.items()}


def subtract_weights(
    updated: Dict[str, np.ndarray], base: Dict[str, np.ndarray]
) -> Dict[str, np.ndarray]:
    return {key: updated[key] - base[key] for key in base}


def add_weights(
    base: Dict[str, np.ndarray], delta: Dict[str, np.ndarray]
) -> Dict[str, np.ndarray]:
    return {key: base[key] + delta[key] for key in base}


def average_deltas(
    deltas: Sequence[Dict[str, np.ndarray]],
) -> Dict[str, np.ndarray]:
    if not deltas:
        return {}
    return {
        key: np.mean([delta[key] for delta in deltas], axis=0).astype(np.float32)
        for key in deltas[0]
    }


@dataclass(frozen=True)
class RoundSummary:
    round_index: int
    mean_latency: float
    mean_energy: float
    violation_rate: float
    privacy_match: float
    normalized_hv: float
    communication_bytes: int
    local_train_loss: float


class PaperAlignedFederatedRunner:
    """Private federated training loop with clipped update aggregation."""

    def __init__(self, config: PaperTrainingConfig | None = None):
        self.config = config or paper_training_config()
        self.global_agent = PaperAlignedMNDDPGAgent(
            self.config.agent, seed=self.config.protocol.split_seed
        )
        self.best_score = float("-inf")
        self.best_weights: Dict[str, np.ndarray] | None = None

    def run_rounds(self, rounds: int | None = None) -> List[RoundSummary]:
        rounds = rounds or self.config.federated.rounds
        summaries: List[RoundSummary] = []
        for round_index in range(rounds):
            base_weights = self.global_agent.get_federated_weights()
            clipped_deltas: List[Dict[str, np.ndarray]] = []
            local_losses: List[float] = []
            communication_bytes = 0

            for client_index in range(self.config.federated.num_clients):
                client_seed = self.config.protocol.independent_run_seeds[
                    client_index % len(self.config.protocol.independent_run_seeds)
                ]
                client_agent = PaperAlignedMNDDPGAgent(
                    self.config.agent, seed=client_seed + round_index * 97
                )
                client_agent.set_federated_weights(base_weights)
                env = PaperAlignedEdgeOffloadingEnv(
                    self.config.environment,
                    seed=client_seed + round_index * 97,
                    mode="seen",
                )
                loss = self._run_local_client(client_agent, env)
                local_losses.append(loss)
                client_weights = client_agent.get_federated_weights()
                delta = subtract_weights(client_weights, base_weights)
                clipped = (
                    clip_delta(delta, self.config.federated.clip_norm)
                    if self.config.federated.use_clipping
                    else {key: value.copy() for key, value in delta.items()}
                )
                clipped_deltas.append(clipped)
                communication_bytes += sum(value.nbytes for value in base_weights.values()) * 2

            averaged_delta = average_deltas(clipped_deltas)
            global_weights = add_weights(base_weights, averaged_delta)
            self.global_agent.set_federated_weights(global_weights)

            eval_stats = self.evaluate(
                mode="seen",
                num_steps=self.config.protocol.evaluation_steps,
            )
            score = float(eval_stats.get("normalized_hv", 0.0))
            if score >= self.best_score:
                self.best_score = score
                self.best_weights = self.global_agent.get_federated_weights()
            summaries.append(
                RoundSummary(
                    round_index=round_index + 1,
                    mean_latency=eval_stats["mean_latency"],
                    mean_energy=eval_stats["mean_energy"],
                    violation_rate=eval_stats["violation_rate"],
                    privacy_match=eval_stats["privacy_match"],
                    normalized_hv=eval_stats["normalized_hv"],
                    communication_bytes=communication_bytes,
                    local_train_loss=float(np.mean(local_losses)) if local_losses else 0.0,
                )
            )
        return summaries

    def evaluate(self, mode: str = "seen", num_steps: int = 24) -> Dict[str, float]:
        env = PaperAlignedEdgeOffloadingEnv(
            self.config.environment,
            seed=self.config.protocol.independent_run_seeds[0] + (1000 if mode == "unseen" else 0),
            mode=mode,
        )
        self.global_agent.reset_episode()
        observation = env.reset()
        records: List[Dict[str, object]] = []
        for _ in range(num_steps):
            action = self.global_agent.select_action(observation, training=False)
            next_observation, info = env.step(action)
            self.global_agent.commit_observation(next_observation.vector)
            records.append(info)
            observation = next_observation

        stats = summarize_evaluation_records(
            records,
            hv_seed=self.config.protocol.split_seed + (1000 if mode == "unseen" else 0),
        )
        stats["records"] = records
        return stats

    def get_best_weights(self) -> Dict[str, np.ndarray]:
        return self.best_weights or self.global_agent.get_federated_weights()

    def _run_local_client(
        self,
        agent: PaperAlignedMNDDPGAgent,
        env: PaperAlignedEdgeOffloadingEnv,
    ) -> float:
        agent.reset_episode()
        observation = env.reset()
        train_losses: List[float] = []

        for step in range(self.config.federated.local_steps):
            action = agent.select_action(observation, training=True)
            next_observation, info = env.step(action)
            next_sequence = agent.preview_sequence(next_observation.vector)
            agent.store_transition(
                observation=observation,
                action=action,
                reward_vector=info["reward_vector"],
                next_observation=next_observation,
                next_sequence_obs=next_sequence,
                done=False,
            )
            agent.commit_observation(next_observation.vector)
            observation = next_observation

            if step >= self.config.agent.batch_size and step % max(
                1, self.config.federated.local_steps // self.config.federated.local_updates
            ) == 0:
                losses = agent.train_step()
                train_losses.append(
                    losses["critic_loss"] + losses["actor_loss"] + losses["proxy_loss"]
                )

        return float(np.mean(train_losses)) if train_losses else 0.0
