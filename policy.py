from __future__ import annotations

from typing import Dict

import numpy as np

try:
    from config import EXPERT_NAMES, PolicyConfig
    from schemas import ActionDecision, Observation
except ImportError:  # pragma: no cover
    from .config import EXPERT_NAMES, PolicyConfig
    from .schemas import ActionDecision, Observation


class MNDDPGReferencePolicy:
    """
    Clean hard-routed policy scaffold.

    This public version keeps the architectural responsibilities from the paper:
    shared features, context-conditioned proxy weights, gating, route masking,
    and projected continuous resources.
    """

    def __init__(self, config: PolicyConfig, seed: int = 0):
        self.config = config
        self.rng = np.random.default_rng(seed)
        scale = config.parameter_scale
        self.shared_w = self.rng.normal(
            0.0, scale, (config.observation_dim, config.shared_dim)
        )
        self.shared_b = np.zeros(config.shared_dim, dtype=np.float64)
        joint_dim = config.shared_dim + config.context_dim
        self.gating_w = self.rng.normal(0.0, scale, (joint_dim, config.num_experts))
        self.gating_b = np.zeros(config.num_experts, dtype=np.float64)
        self.proxy_w = self.rng.normal(0.0, scale, (config.context_dim, config.num_experts))
        self.proxy_b = np.zeros(config.num_experts, dtype=np.float64)
        self.route_w = self.rng.normal(
            0.0, scale, (config.num_experts, joint_dim, config.num_routes)
        )
        self.route_b = np.zeros((config.num_experts, config.num_routes), dtype=np.float64)
        self.resource_w = self.rng.normal(
            0.0, scale, (config.num_experts, joint_dim, 3)
        )
        self.resource_b = np.zeros((config.num_experts, 3), dtype=np.float64)
        self.reliability_trace = np.zeros(joint_dim, dtype=np.float64)

    def select_action(
        self, observation: Observation, training: bool = False
    ) -> ActionDecision:
        features = np.tanh(observation.vector @ self.shared_w + self.shared_b)
        joint = np.concatenate([features, observation.context])

        expert_logits = joint @ self.gating_w + self.gating_b
        expert_logits += self._expert_heuristic_bias(observation)
        expert_probs = self._softmax(expert_logits)
        if training:
            expert_probs = self._gumbel_softmax(
                expert_logits, self.config.gating_temperature
            )
        expert_index = int(np.argmax(expert_probs))

        proxy_logits = observation.context @ self.proxy_w + self.proxy_b
        proxy_weights = self._softmax(proxy_logits)

        expert_joint = joint.copy()
        if expert_index == 2:
            self.reliability_trace = 0.85 * self.reliability_trace + 0.15 * joint
            expert_joint = 0.7 * joint + 0.3 * self.reliability_trace

        route_logits = (
            expert_joint @ self.route_w[expert_index]
            + self.route_b[expert_index]
            + self._route_heuristic_bias(observation, expert_index)
        )
        masked_logits = np.where(
            observation.feasible_mask > 0,
            route_logits,
            -1.0e9,
        )
        if training:
            route_scores = self._gumbel_softmax(
                masked_logits, self.config.route_temperature
            )
        else:
            route_scores = self._softmax(masked_logits)
        route_index = int(np.argmax(route_scores))

        resource_raw = np.tanh(
            expert_joint @ self.resource_w[expert_index] + self.resource_b[expert_index]
        )
        resource_unit = 0.5 * (resource_raw + 1.0)
        resource_unit = 0.4 * resource_unit + 0.6 * self._resource_default(
            observation, expert_index
        )

        return ActionDecision(
            route_index=route_index,
            route_name=observation.candidate_nodes[route_index],
            bandwidth_ratio=float(np.clip(resource_unit[0], 0.05, 1.0)),
            cpu_ratio=float(np.clip(resource_unit[1], 0.05, 1.0)),
            tx_power=float(np.clip(resource_unit[2], 0.0, 1.0)),
            expert_index=expert_index,
            expert_name=EXPERT_NAMES[expert_index],
            expert_probs=expert_probs,
            proxy_weights=proxy_weights,
        )

    def clone(self) -> "MNDDPGReferencePolicy":
        clone = MNDDPGReferencePolicy(self.config, seed=0)
        clone.set_parameters(self.get_parameters())
        clone.reliability_trace = self.reliability_trace.copy()
        return clone

    def get_parameters(self) -> Dict[str, np.ndarray]:
        return {
            "shared_w": self.shared_w.copy(),
            "shared_b": self.shared_b.copy(),
            "gating_w": self.gating_w.copy(),
            "gating_b": self.gating_b.copy(),
            "proxy_w": self.proxy_w.copy(),
            "proxy_b": self.proxy_b.copy(),
            "route_w": self.route_w.copy(),
            "route_b": self.route_b.copy(),
            "resource_w": self.resource_w.copy(),
            "resource_b": self.resource_b.copy(),
        }

    def set_parameters(self, params: Dict[str, np.ndarray]) -> None:
        self.shared_w = params["shared_w"].copy()
        self.shared_b = params["shared_b"].copy()
        self.gating_w = params["gating_w"].copy()
        self.gating_b = params["gating_b"].copy()
        self.proxy_w = params["proxy_w"].copy()
        self.proxy_b = params["proxy_b"].copy()
        self.route_w = params["route_w"].copy()
        self.route_b = params["route_b"].copy()
        self.resource_w = params["resource_w"].copy()
        self.resource_b = params["resource_b"].copy()

    @staticmethod
    def _softmax(logits: np.ndarray) -> np.ndarray:
        safe = logits - np.max(logits)
        exp = np.exp(safe)
        return exp / np.sum(exp)

    def _gumbel_softmax(self, logits: np.ndarray, temperature: float) -> np.ndarray:
        noise = -np.log(-np.log(self.rng.uniform(1e-8, 1.0 - 1e-8, logits.shape)))
        return self._softmax((logits + noise) / max(temperature, 1e-6))

    def _expert_heuristic_bias(self, observation: Observation) -> np.ndarray:
        queue_pressure = float(observation.context[1])
        energy_pressure = float(1.0 - observation.context[4])
        reliability_pressure = float(max(observation.context[3], 1.0 - observation.context[7]))
        privacy_pressure = float(observation.current_task.privacy_level / 10.0)
        deadline_urgency = float(observation.context[8])
        return np.array(
            [
                1.6 * deadline_urgency + 0.5 * queue_pressure,
                1.5 * energy_pressure,
                1.4 * reliability_pressure + 0.3 * queue_pressure,
                1.8 * privacy_pressure + 0.6 * observation.context[9],
            ],
            dtype=np.float64,
        )

    def _route_heuristic_bias(
        self, observation: Observation, expert_index: int
    ) -> np.ndarray:
        bias = np.zeros(self.config.num_routes, dtype=np.float64)
        local_index = 0
        cloud_index = self.config.num_routes - 1
        edge_slice = slice(1, cloud_index)
        privacy_pressure = observation.current_task.privacy_level / 10.0
        compute_pressure = float(observation.vector[10])
        data_pressure = float(observation.vector[9])
        deadline_urgency = float(observation.context[8])

        if expert_index == 0:
            bias[edge_slice] += 0.9 + 0.8 * deadline_urgency
            bias[local_index] += 0.25 + 0.4 * privacy_pressure
            bias[cloud_index] += 0.2 + 0.2 * compute_pressure - 0.8 * deadline_urgency
        elif expert_index == 1:
            bias[local_index] += 0.5 + 0.3 * (1.0 - data_pressure)
            bias[edge_slice] += 0.55
            bias[cloud_index] -= 0.2
        elif expert_index == 2:
            bias[local_index] += 0.45
            bias[edge_slice] += 0.8
            bias[cloud_index] += 0.15
        else:
            bias[local_index] += 1.2 * privacy_pressure
            bias[edge_slice] += 0.6 * privacy_pressure
            bias[cloud_index] -= 1.0 * privacy_pressure

        if compute_pressure > 0.7 and privacy_pressure < 0.6:
            bias[cloud_index] += 0.6
        return bias

    def _resource_default(
        self, observation: Observation, expert_index: int
    ) -> np.ndarray:
        deadline_urgency = float(observation.context[8])
        privacy_pressure = float(observation.current_task.privacy_level / 10.0)
        presets = {
            0: np.array([0.82, 0.80, 0.82]),
            1: np.array([0.45, 0.58, 0.32]),
            2: np.array([0.68, 0.78, 0.54]),
            3: np.array([0.52, 0.66, 0.24]),
        }
        default = presets[expert_index].copy()
        default[0] = np.clip(default[0] + 0.10 * deadline_urgency, 0.05, 1.0)
        if privacy_pressure > 0.7:
            default[2] = np.clip(default[2] - 0.12, 0.0, 1.0)
        return default
