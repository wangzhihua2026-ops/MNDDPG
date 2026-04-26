from __future__ import annotations

from collections import deque
from typing import Dict, List, Tuple

import numpy as np

try:
    from config import EnvironmentConfig
    from schemas import ActionDecision, NodeSpec, Observation, TaskSpec
except ImportError:  # pragma: no cover
    from .config import EnvironmentConfig
    from .schemas import ActionDecision, NodeSpec, Observation, TaskSpec


class EdgeOffloadingEnv:
    """Paper-aligned public environment for privacy-aware edge offloading."""

    def __init__(self, config: EnvironmentConfig, seed: int):
        self.config = config
        self.seed = seed
        self.rng = np.random.default_rng(seed)
        self.current_time = 0
        self.task_counter = 0
        self.recent_privacy_levels: deque[int] = deque(
            maxlen=config.recent_sensitivity_window
        )
        self.nodes: List[NodeSpec] = []
        self.current_task: TaskSpec | None = None
        self.reset()

    def reset(self) -> Observation:
        self.current_time = 0
        self.task_counter = 0
        self.recent_privacy_levels.clear()
        self.nodes = [self._build_local_node()]
        self.nodes.extend(self._build_edge_nodes())
        if self.config.include_cloud:
            self.nodes.append(self._build_cloud_node())
        self.current_task = self._sample_task()
        return self._build_observation()

    def step(self, action: ActionDecision) -> Tuple[Observation, Dict[str, object]]:
        task = self.current_task
        if task is None:
            raise RuntimeError("Environment has no active task.")

        target = self.nodes[action.route_index]
        bandwidth_ratio = float(np.clip(action.bandwidth_ratio, 0.05, 1.0))
        cpu_ratio = float(np.clip(action.cpu_ratio, 0.05, 1.0))
        tx_power = float(np.clip(action.tx_power, 0.0, task.max_tx_power))

        transfer_delay = 0.0
        if target.kind != "local":
            rate = self._effective_rate(target, bandwidth_ratio)
            propagation = (
                self.config.propagation_delay_cloud
                if target.kind == "cloud"
                else self.config.propagation_delay_edge
            )
            transfer_delay = task.data_size / max(rate, 1e-6) + propagation

        queue_delay = target.queue_backlog
        exec_delay = (
            task.compute_demand / max(target.compute_capacity * cpu_ratio, 1e-6)
            + task.data_size / max(target.io_rate, 1e-6)
        )
        total_delay = transfer_delay + queue_delay + exec_delay
        total_energy = self._compute_energy(
            task=task,
            target=target,
            exec_delay=exec_delay,
            transfer_delay=transfer_delay,
            cpu_ratio=cpu_ratio,
            tx_power=tx_power,
        )
        privacy_score = self._privacy_score(task, target)
        reliability_ok = (
            total_delay <= task.deadline
            and target.reliability >= self.config.reliability_threshold
        )

        reward_vector = np.array(
            [
                max(0.0, 1.0 - total_delay / max(task.deadline, 1e-6)),
                max(
                    0.0,
                    1.0 - total_energy / max(self.config.energy_budget_max, 1e-6),
                ),
                1.0 if reliability_ok else 0.0,
                privacy_score,
            ],
            dtype=np.float64,
        )
        cost_vector = np.array(
            [
                total_delay,
                total_energy,
                0.0 if reliability_ok else 1.0,
                1.0 - privacy_score,
            ],
            dtype=np.float64,
        )
        privacy_match = bool(target.security_level >= task.privacy_level)
        violation = bool(
            (not reliability_ok)
            or (task.sensitivity == "high" and not privacy_match)
        )

        self._advance_system(
            target_index=action.route_index,
            exec_delay=exec_delay,
            local_energy=total_energy,
        )

        info: Dict[str, object] = {
            "task_id": task.task_id,
            "profile": task.profile,
            "privacy_level": task.privacy_level,
            "selected_node": target.node_id,
            "selected_node_index": action.route_index,
            "reward_vector": reward_vector,
            "proxy_reward": float(np.dot(action.proxy_weights, reward_vector)),
            "cost_vector": cost_vector,
            "latency": float(total_delay),
            "energy": float(total_energy),
            "privacy_match": privacy_match,
            "violation": violation,
            "expert_name": action.expert_name,
            "expert_index": action.expert_index,
        }

        self.current_task = self._sample_task()
        return self._build_observation(), info

    def trusted_route_mask(self) -> np.ndarray:
        return np.array(
            [1.0 if node.kind == "local" or node.trusted else 0.0 for node in self.nodes],
            dtype=np.float64,
        )

    def _build_local_node(self) -> NodeSpec:
        return NodeSpec(
            node_id="local_device",
            kind="local",
            compute_capacity=float(self.rng.uniform(70.0, 100.0)),
            memory_capacity=float(self.rng.uniform(4.0, 8.0)),
            io_rate=float(self.rng.uniform(85.0, 120.0)),
            security_level=self.config.privacy_levels,
            base_power=0.018,
            load_amp=0.55,
            reliability=float(self.rng.uniform(0.93, 0.99)),
            available_bandwidth=float(self.rng.uniform(40.0, 70.0)),
            remaining_energy=1.0,
            trusted=True,
        )

    def _build_edge_nodes(self) -> List[NodeSpec]:
        nodes: List[NodeSpec] = []
        for index in range(self.config.num_edge_nodes):
            nodes.append(
                NodeSpec(
                    node_id=f"edge_{index}",
                    kind="edge",
                    compute_capacity=float(self.rng.uniform(150.0, 260.0)),
                    memory_capacity=float(self.rng.uniform(12.0, 32.0)),
                    io_rate=float(self.rng.uniform(160.0, 260.0)),
                    security_level=int(
                        self.rng.integers(4, self.config.privacy_levels + 1)
                    ),
                    base_power=float(self.rng.uniform(0.05, 0.12)),
                    load_amp=float(self.rng.uniform(0.25, 0.60)),
                    reliability=float(self.rng.uniform(0.84, 0.98)),
                    available_bandwidth=float(self.rng.uniform(50.0, 95.0)),
                    trusted=index in self.config.trusted_edge_indices,
                )
            )
        return nodes

    def _build_cloud_node(self) -> NodeSpec:
        return NodeSpec(
            node_id="cloud",
            kind="cloud",
            compute_capacity=float(self.rng.uniform(320.0, 460.0)),
            memory_capacity=128.0,
            io_rate=float(self.rng.uniform(300.0, 420.0)),
            security_level=max(4, self.config.high_privacy_threshold - 1),
            base_power=0.16,
            load_amp=0.25,
            reliability=float(self.rng.uniform(0.95, 0.995)),
            available_bandwidth=float(self.rng.uniform(70.0, 120.0)),
        )

    def _sample_task(self) -> TaskSpec:
        profile = self.rng.choice(
            self.config.task_profiles,
            p=[profile.weight for profile in self.config.task_profiles],
        )
        privacy_draw = float(self.rng.random())
        if privacy_draw < self.config.low_privacy_ratio:
            privacy_level = int(
                self.rng.integers(1, self.config.medium_privacy_threshold)
            )
        elif privacy_draw < self.config.low_privacy_ratio + self.config.medium_privacy_ratio:
            privacy_level = int(
                self.rng.integers(
                    self.config.medium_privacy_threshold,
                    self.config.high_privacy_threshold,
                )
            )
        else:
            privacy_level = int(
                self.rng.integers(
                    self.config.high_privacy_threshold,
                    self.config.privacy_levels + 1,
                )
            )

        task = TaskSpec(
            task_id=f"task_{self.seed}_{self.task_counter}",
            source_device="local_device",
            profile=profile.name,
            data_size=self._sample_truncated_lognormal(
                profile.data_mu, profile.data_sigma, *profile.data_bounds
            ),
            compute_demand=self._sample_truncated_lognormal(
                profile.compute_mu, profile.compute_sigma, *profile.compute_bounds
            ),
            memory_demand=float(self.rng.uniform(*profile.memory_range)),
            deadline=float(self.rng.uniform(*profile.deadline_range)),
            priority=int(
                self.rng.integers(
                    profile.priority_range[0], profile.priority_range[1] + 1
                )
            ),
            privacy_level=privacy_level,
            max_tx_power=self.config.max_tx_power,
        )
        self.task_counter += 1
        self.recent_privacy_levels.append(privacy_level)
        return task

    def _sample_truncated_lognormal(
        self, mu: float, sigma: float, lower: float, upper: float
    ) -> float:
        sample = lower
        for _ in range(20):
            sample = float(self.rng.lognormal(mu, sigma))
            if lower <= sample <= upper:
                return sample
        return float(np.clip(sample, lower, upper))

    def _build_observation(self) -> Observation:
        task = self.current_task
        if task is None:
            raise RuntimeError("Environment has no active task.")

        local = self.nodes[0]
        neighbors = self.nodes[1:]
        avg_neighbor_cpu = (
            float(np.mean([node.cpu_utilization for node in neighbors]))
            if neighbors
            else 0.0
        )
        avg_neighbor_queue = (
            float(np.mean([node.queue_backlog for node in neighbors]))
            if neighbors
            else 0.0
        )
        avg_neighbor_bw = (
            float(np.mean([node.available_bandwidth for node in neighbors]))
            if neighbors
            else 0.0
        )
        avg_neighbor_rel = (
            float(np.mean([node.reliability for node in neighbors]))
            if neighbors
            else 1.0
        )
        packet_loss = float(
            np.clip(
                self.config.packet_loss_base
                + self.rng.normal(0.0, self.config.rate_noise_std * 0.08),
                0.0,
                0.35,
            )
        )
        max_data = max(profile.data_bounds[1] for profile in self.config.task_profiles)
        max_compute = max(
            profile.compute_bounds[1] for profile in self.config.task_profiles
        )
        max_deadline = max(
            profile.deadline_range[1] for profile in self.config.task_profiles
        )

        vector = np.array(
            [
                local.cpu_utilization,
                local.available_bandwidth / self.config.max_uplink_rate,
                min(1.0, local.queue_backlog / self.config.queue_scale),
                local.remaining_energy,
                packet_loss,
                avg_neighbor_cpu,
                min(1.0, avg_neighbor_queue / self.config.queue_scale),
                avg_neighbor_bw / self.config.max_uplink_rate,
                avg_neighbor_rel,
                task.data_size / max_data,
                task.compute_demand / max_compute,
                task.deadline / max_deadline,
                task.priority / 10.0,
                task.privacy_level / self.config.privacy_levels,
            ],
            dtype=np.float64,
        )

        medium_or_high = [
            1
            for level in self.recent_privacy_levels
            if level >= self.config.medium_privacy_threshold
        ]
        sensitivity_ratio = float(
            sum(medium_or_high) / max(len(self.recent_privacy_levels), 1)
        )
        context = np.array(
            [
                local.cpu_utilization,
                min(1.0, local.queue_backlog / self.config.queue_scale),
                local.available_bandwidth / self.config.max_uplink_rate,
                packet_loss,
                local.remaining_energy,
                avg_neighbor_cpu,
                min(1.0, avg_neighbor_queue / self.config.queue_scale),
                avg_neighbor_rel,
                1.0 - vector[11],
                sensitivity_ratio,
            ],
            dtype=np.float64,
        )

        feasible_mask = self._feasible_mask(task)
        return Observation(
            vector=vector,
            context=context,
            feasible_mask=feasible_mask,
            current_task=task,
            candidate_nodes=[node.node_id for node in self.nodes],
        )

    def _feasible_mask(self, task: TaskSpec) -> np.ndarray:
        mask = []
        for node in self.nodes:
            memory_ok = task.memory_demand <= node.memory_capacity
            privacy_ok = True
            if task.privacy_level >= self.config.high_privacy_threshold:
                privacy_ok = node.kind == "local" or node.trusted
            mask.append(1.0 if memory_ok and privacy_ok else 0.0)
        return np.array(mask, dtype=np.float64)

    def _effective_rate(self, target: NodeSpec, bandwidth_ratio: float) -> float:
        local_rate = self.nodes[0].available_bandwidth
        congestion = np.clip(target.cpu_utilization, 0.0, 1.0)
        noise = float(self.rng.normal(0.0, self.config.rate_noise_std))
        rate = (
            local_rate
            * bandwidth_ratio
            * (1.0 - self.config.congestion_sensitivity * congestion)
            * (1.0 + noise)
        )
        return float(
            np.clip(
                rate,
                self.config.min_uplink_rate,
                self.config.max_uplink_rate,
            )
        )

    def _compute_energy(
        self,
        task: TaskSpec,
        target: NodeSpec,
        exec_delay: float,
        transfer_delay: float,
        cpu_ratio: float,
        tx_power: float,
    ) -> float:
        if target.kind == "local":
            freq = self.config.local_cpu_frequency * cpu_ratio
            return float(self.config.local_dvfs_kappa * (freq ** 2) * task.compute_demand)

        comm_energy = tx_power * transfer_delay
        remote_energy = (
            target.base_power * exec_delay * (1.0 + target.load_amp * target.cpu_utilization)
        )
        return float(comm_energy + remote_energy)

    def _privacy_score(self, task: TaskSpec, target: NodeSpec) -> float:
        gap = max(0, task.privacy_level - target.security_level)
        return float(1.0 - gap / max(self.config.privacy_levels - 1, 1))

    def _advance_system(self, target_index: int, exec_delay: float, local_energy: float) -> None:
        self.current_time += 1
        for index, node in enumerate(self.nodes):
            node.queue_backlog = max(0.0, node.queue_backlog - self.config.queue_decay_per_step)
            if index == target_index:
                node.queue_backlog += exec_delay
            node.cpu_utilization = float(
                np.clip(node.queue_backlog / self.config.queue_scale, 0.0, 1.0)
            )
            node.available_bandwidth = float(
                np.clip(
                    node.available_bandwidth + self.rng.normal(0.0, 4.0),
                    self.config.min_uplink_rate,
                    self.config.max_uplink_rate + 30.0,
                )
            )
            node.reliability = float(
                np.clip(
                    node.reliability
                    - 0.015 * node.cpu_utilization
                    + self.rng.normal(0.0, 0.004),
                    0.75,
                    0.995,
                )
            )

        self.nodes[0].remaining_energy = float(
            np.clip(
                self.nodes[0].remaining_energy
                - local_energy / (self.config.energy_budget_max * 20.0),
                0.0,
                1.0,
            )
        )
