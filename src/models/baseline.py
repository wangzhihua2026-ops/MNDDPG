from __future__ import annotations

import numpy as np

from src.data.schemas import ActionRecord, Observation


class _BaseFeasiblePolicy:
    expert_index = 0

    def _feasible_indices(self, observation: Observation) -> np.ndarray:
        feasible_indices = np.flatnonzero(observation.feasible_mask > 0)
        if feasible_indices.size == 0:
            raise ValueError("Observation has no feasible route.")
        return feasible_indices

    def _build_action(
        self,
        observation: Observation,
        route_index: int,
        continuous_action: np.ndarray,
    ) -> ActionRecord:
        if observation.feasible_mask[route_index] <= 0:
            route_index = int(self._feasible_indices(observation)[0])
        route_onehot = np.zeros_like(observation.feasible_mask, dtype=np.float32)
        route_onehot[route_index] = 1.0
        expert_probs = np.full(4, 0.25, dtype=np.float32)
        proxy_weights = np.full(4, 0.25, dtype=np.float32)
        return ActionRecord(
            route_index=route_index,
            route_onehot=route_onehot,
            continuous_action=continuous_action.astype(np.float32),
            expert_index=self.expert_index,
            expert_probs=expert_probs,
            proxy_weights=proxy_weights,
            sequence_obs=np.zeros((1, observation.vector.shape[0]), dtype=np.float32),
        )


class RandomFeasiblePolicy:
    """Simple sanity-check policy that samples one currently feasible route."""

    def __init__(self, seed: int = 0):
        self.rng = np.random.default_rng(seed)

    def select_action(self, observation: Observation) -> ActionRecord:
        feasible_indices = np.flatnonzero(observation.feasible_mask > 0)
        if feasible_indices.size == 0:
            raise ValueError("Observation has no feasible route.")

        route_index = int(self.rng.choice(feasible_indices))
        continuous_action = self.rng.uniform(0.2, 0.8, size=3).astype(np.float32)
        return _BaseFeasiblePolicy()._build_action(observation, route_index, continuous_action)


class LocalOnlyPolicy(_BaseFeasiblePolicy):
    """Always select the local source-device slot when it is feasible."""

    def select_action(self, observation: Observation) -> ActionRecord:
        return self._build_action(
            observation,
            route_index=0,
            continuous_action=np.array([0.05, 0.65, 0.0], dtype=np.float32),
        )


class EdgeOnlyPolicy(_BaseFeasiblePolicy):
    """Select the first feasible edge node and avoid cloud when possible."""

    def select_action(self, observation: Observation) -> ActionRecord:
        edge_indices = [
            index
            for index, name in enumerate(observation.candidate_nodes)
            if name.startswith("edge_") and observation.feasible_mask[index] > 0
        ]
        route_index = edge_indices[0] if edge_indices else int(self._feasible_indices(observation)[0])
        return self._build_action(
            observation,
            route_index=route_index,
            continuous_action=np.array([0.6, 0.7, 0.45], dtype=np.float32),
        )


class GreedyMinLatencyPolicy(_BaseFeasiblePolicy):
    """Latency-oriented heuristic with aggressive resource allocation."""

    expert_index = 0

    def select_action(self, observation: Observation) -> ActionRecord:
        feasible = self._feasible_indices(observation)
        edge_or_cloud = [
            index
            for index in feasible
            if observation.candidate_nodes[index].startswith("edge_")
            or observation.candidate_nodes[index] == "cloud"
        ]
        route_index = edge_or_cloud[0] if edge_or_cloud else int(feasible[0])
        return self._build_action(
            observation,
            route_index=route_index,
            continuous_action=np.array([0.95, 0.95, 0.9], dtype=np.float32),
        )


class GreedyMinEnergyPolicy(_BaseFeasiblePolicy):
    """Energy-oriented heuristic with conservative transmit power."""

    expert_index = 1

    def select_action(self, observation: Observation) -> ActionRecord:
        feasible = self._feasible_indices(observation)
        route_index = 0 if observation.feasible_mask[0] > 0 else int(feasible[0])
        return self._build_action(
            observation,
            route_index=route_index,
            continuous_action=np.array([0.25, 0.45, 0.05], dtype=np.float32),
        )


def available_baseline_policies(seed: int = 0) -> dict[str, object]:
    return {
        "RandomFeasible": RandomFeasiblePolicy(seed=seed),
        "LocalOnly": LocalOnlyPolicy(),
        "EdgeOnly": EdgeOnlyPolicy(),
        "GreedyMinLatency": GreedyMinLatencyPolicy(),
        "GreedyMinEnergy": GreedyMinEnergyPolicy(),
    }
