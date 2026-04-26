from __future__ import annotations

from typing import Dict, List

import numpy as np

try:
    from config import ExperimentConfig, paper_reference_experiment
    from environment import EdgeOffloadingEnv
    from federated import ModelDelta, aggregate_deltas
    from metrics import summarize_records
    from policy import MNDDPGReferencePolicy
except ImportError:  # pragma: no cover
    from .config import ExperimentConfig, paper_reference_experiment
    from .environment import EdgeOffloadingEnv
    from .federated import ModelDelta, aggregate_deltas
    from .metrics import summarize_records
    from .policy import MNDDPGReferencePolicy


class PublicExperimentRunner:
    """
    Public demo runner.

    The update rule is intentionally lightweight and transparent so the public
    release stays auditable. Replace `_build_surrogate_local_delta` with the
    full private actor-critic training loop if you need full research fidelity.
    """

    def __init__(self, config: ExperimentConfig | None = None):
        self.config = config or paper_reference_experiment()
        self.global_policy = MNDDPGReferencePolicy(
            self.config.policy, seed=self.config.split_seed
        )

    def run(self) -> Dict[str, object]:
        round_summaries: List[Dict[str, object]] = []
        cumulative_comm = 0

        for round_index in range(self.config.rounds):
            base_params = self.global_policy.get_parameters()
            local_deltas: List[ModelDelta] = []

            for client_index in range(self.config.federated.num_clients):
                client_seed = self.config.train_client_seeds[
                    client_index % len(self.config.train_client_seeds)
                ]
                env = EdgeOffloadingEnv(
                    self.config.environment, seed=client_seed + round_index * 97
                )
                local_policy = self.global_policy.clone()
                observation = env.reset()
                client_records: List[Dict[str, object]] = []

                for _ in range(self.config.steps_per_round):
                    action = local_policy.select_action(observation, training=False)
                    observation, info = env.step(action)
                    client_records.append(info)

                delta = self._build_surrogate_local_delta(
                    base_params, client_records, env
                )
                local_deltas.append(
                    ModelDelta(delta=delta, num_samples=len(client_records))
                )

            updated_params, round_comm = aggregate_deltas(
                base_params, local_deltas, self.config.federated.clip_norm
            )
            cumulative_comm += round_comm
            self.global_policy.set_parameters(updated_params)

            eval_records = self._evaluate_current_policy()
            summary = summarize_records(eval_records)
            summary["round"] = round_index + 1
            summary["round_communication_bytes"] = round_comm
            summary["cumulative_communication_bytes"] = cumulative_comm
            round_summaries.append(summary)

        headline = round_summaries[-1] if round_summaries else {}
        return {
            "config": {
                "rounds": self.config.rounds,
                "steps_per_round": self.config.steps_per_round,
                "num_clients": self.config.federated.num_clients,
                "eval_scenarios": len(self.config.eval_scenario_seeds),
            },
            "headline": headline,
            "rounds": round_summaries,
        }

    def _evaluate_current_policy(self) -> List[Dict[str, object]]:
        records: List[Dict[str, object]] = []
        for seed in self.config.eval_scenario_seeds:
            env = EdgeOffloadingEnv(self.config.environment, seed=seed)
            observation = env.reset()
            for _ in range(self.config.steps_per_round):
                action = self.global_policy.select_action(observation, training=False)
                observation, info = env.step(action)
                info["scenario_seed"] = seed
                records.append(info)
        return records

    def _build_surrogate_local_delta(
        self,
        base_params: Dict[str, np.ndarray],
        client_records: List[Dict[str, object]],
        env: EdgeOffloadingEnv,
    ) -> Dict[str, np.ndarray]:
        delta = {key: np.zeros_like(value) for key, value in base_params.items()}
        if not client_records:
            return delta

        rewards = np.array(
            [record["reward_vector"] for record in client_records],
            dtype=np.float64,
        )
        mean_rewards = rewards.mean(axis=0)
        bottleneck = 1.0 - mean_rewards
        centered_bottleneck = bottleneck - np.mean(bottleneck)
        privacy_pressure = float(
            np.mean(
                [
                    int(record["privacy_level"])
                    >= env.config.medium_privacy_threshold
                    for record in client_records
                ]
            )
        )

        lr = 0.045
        delta["proxy_b"] += lr * centered_bottleneck
        delta["gating_b"] += lr * centered_bottleneck

        trusted_mask = env.trusted_route_mask()
        untrusted_mask = 1.0 - trusted_mask
        delta["route_b"][3] += (
            lr * (trusted_mask - 0.35 * untrusted_mask) * max(privacy_pressure, 0.15)
        )
        delta["route_b"][0][0] += lr * bottleneck[0]
        delta["route_b"][1][0] -= lr * bottleneck[1] * 0.5

        delta["resource_b"][0] += np.array([0.10, 0.08, 0.10]) * bottleneck[0] * lr
        delta["resource_b"][1] += np.array([-0.08, -0.06, -0.10]) * bottleneck[1] * lr
        delta["resource_b"][2] += np.array([0.02, 0.12, 0.04]) * bottleneck[2] * lr
        delta["resource_b"][3] += np.array([-0.02, 0.03, -0.06]) * privacy_pressure * lr

        return delta
