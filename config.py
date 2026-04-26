from __future__ import annotations

from dataclasses import dataclass, field
from typing import Tuple

NUM_OBJECTIVES = 4
EXPERT_NAMES = ("latency", "energy", "reliability", "privacy")


@dataclass(frozen=True)
class TaskProfileConfig:
    name: str
    weight: float
    compute_mu: float
    compute_sigma: float
    compute_bounds: Tuple[float, float]
    data_mu: float
    data_sigma: float
    data_bounds: Tuple[float, float]
    memory_range: Tuple[float, float]
    deadline_range: Tuple[float, float]
    priority_range: Tuple[int, int]


@dataclass(frozen=True)
class EnvironmentConfig:
    num_edge_nodes: int = 6
    include_cloud: bool = True
    privacy_levels: int = 10
    medium_privacy_threshold: int = 5
    high_privacy_threshold: int = 8
    reliability_threshold: float = 0.86
    max_tx_power: float = 1.0
    max_uplink_rate: float = 90.0
    min_uplink_rate: float = 12.0
    congestion_sensitivity: float = 0.35
    rate_noise_std: float = 0.12
    packet_loss_base: float = 0.04
    propagation_delay_edge: float = 0.02
    propagation_delay_cloud: float = 0.06
    energy_budget_max: float = 6.0
    queue_decay_per_step: float = 1.0
    queue_scale: float = 5.0
    local_dvfs_kappa: float = 0.0035
    local_cpu_frequency: float = 2.4
    recent_sensitivity_window: int = 20
    low_privacy_ratio: float = 0.33
    medium_privacy_ratio: float = 0.34
    high_privacy_ratio: float = 0.33
    trusted_edge_indices: Tuple[int, ...] = (0, 1, 2)
    task_profiles: Tuple[TaskProfileConfig, ...] = field(
        default_factory=lambda: (
            TaskProfileConfig(
                name="compute_intensive",
                weight=0.30,
                compute_mu=4.7,
                compute_sigma=0.30,
                compute_bounds=(70.0, 220.0),
                data_mu=2.7,
                data_sigma=0.25,
                data_bounds=(8.0, 36.0),
                memory_range=(1.2, 4.5),
                deadline_range=(2.0, 4.0),
                priority_range=(5, 9),
            ),
            TaskProfileConfig(
                name="data_intensive",
                weight=0.40,
                compute_mu=3.9,
                compute_sigma=0.25,
                compute_bounds=(35.0, 130.0),
                data_mu=4.3,
                data_sigma=0.30,
                data_bounds=(35.0, 160.0),
                memory_range=(0.8, 3.2),
                deadline_range=(3.0, 6.5),
                priority_range=(3, 8),
            ),
            TaskProfileConfig(
                name="realtime_sensitive",
                weight=0.30,
                compute_mu=4.1,
                compute_sigma=0.25,
                compute_bounds=(45.0, 145.0),
                data_mu=2.5,
                data_sigma=0.20,
                data_bounds=(6.0, 24.0),
                memory_range=(0.6, 2.4),
                deadline_range=(0.8, 1.6),
                priority_range=(7, 10),
            ),
        )
    )

    @property
    def num_routes(self) -> int:
        return 1 + self.num_edge_nodes + (1 if self.include_cloud else 0)

    @property
    def observation_dim(self) -> int:
        return 14

    @property
    def context_dim(self) -> int:
        return 10


@dataclass(frozen=True)
class PolicyConfig:
    observation_dim: int = 14
    context_dim: int = 10
    shared_dim: int = 24
    num_experts: int = 4
    num_routes: int = 8
    gating_temperature: float = 0.8
    route_temperature: float = 0.45
    parameter_scale: float = 0.16


@dataclass(frozen=True)
class FederatedConfig:
    num_clients: int = 6
    clip_norm: float = 1.75
    secure_aggregation: bool = True


@dataclass(frozen=True)
class ExperimentConfig:
    environment: EnvironmentConfig
    policy: PolicyConfig
    federated: FederatedConfig
    split_seed: int = 3407
    train_client_seeds: Tuple[int, ...] = (2026, 2027, 2028, 2029, 2030, 2031)
    eval_scenario_seeds: Tuple[int, ...] = tuple(range(3001, 3013))
    rounds: int = 4
    steps_per_round: int = 32


def paper_reference_experiment() -> ExperimentConfig:
    env = EnvironmentConfig()
    policy = PolicyConfig(
        observation_dim=env.observation_dim,
        context_dim=env.context_dim,
        num_routes=env.num_routes,
    )
    return ExperimentConfig(
        environment=env,
        policy=policy,
        federated=FederatedConfig(num_clients=6),
    )
