from __future__ import annotations

from collections import deque
from typing import Dict, List, Tuple

import numpy as np

from src.data.schemas import ActionRecord, NodeSpec, Observation, TaskSpec
from src.utils.paper_config import EnvironmentConfig


class PaperAlignedEdgeOffloadingEnv:
    """Environment aligned with the problem formulation in main.tex."""

    def __init__(self, config: EnvironmentConfig, seed: int, mode: str = "seen"):
        self.config = config
        self.seed = seed
        self.mode = mode
        self.rng = np.random.default_rng(seed)
        self.current_time = 0
        self.task_counter = 0
        self.mmpp_state = 1
        self.recent_privacy_levels: deque[int] = deque(
            maxlen=config.recent_sensitivity_window
        )
        self.pending_tasks: deque[TaskSpec] = deque()
        self.nodes: List[NodeSpec] = []
        self.current_task: TaskSpec | None = None
        self._build_distribution_context()
        self.reset()

    def reset(self) -> Observation:
        self.current_time = 0
        self.task_counter = 0
        self.mmpp_state = 1
        self.recent_privacy_levels.clear()
        self.pending_tasks.clear()
        self.nodes = [self._build_local_node()]
        self.nodes.extend(self._build_edge_nodes())
        if self.config.include_cloud:
            self.nodes.append(self._build_cloud_node())
        self._enqueue_arrivals(force_non_empty=True)
        self.current_task = self.pending_tasks.popleft()
        return self._build_observation()

    def step(self, action: ActionRecord) -> Tuple[Observation, Dict[str, object]]:
        task = self.current_task
        if task is None:
            raise RuntimeError("Environment has no active task.")

        target = self.nodes[action.route_index]
        bandwidth_ratio = float(np.clip(action.continuous_action[0], 0.05, 1.0))
        cpu_ratio = float(np.clip(action.continuous_action[1], 0.05, 1.0))
        tx_power = float(np.clip(action.continuous_action[2], 0.0, task.max_tx_power))

        transmission_delay = 0.0
        if target.kind != "local":
            uplink_rate = self._effective_rate(target, bandwidth_ratio)
            propagation = (
                self.config.propagation_delay_cloud
                if target.kind == "cloud"
                else self.config.propagation_delay_edge
            )
            transmission_delay = task.data_size / max(uplink_rate, 1e-6) + propagation

        queue_delay = target.queue_backlog
        execution_delay = (
            task.compute_demand / max(target.compute_capacity * cpu_ratio, 1e-6)
            + task.data_size / max(target.io_rate, 1e-6)
        )
        total_delay = transmission_delay + queue_delay + execution_delay

        local_energy, total_energy = self._compute_energy(
            task=task,
            target=target,
            execution_delay=execution_delay,
            transmission_delay=transmission_delay,
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
                max(0.0, 1.0 - total_energy / max(self.config.energy_budget_max, 1e-6)),
                1.0 if reliability_ok else 0.0,
                privacy_score,
            ],
            dtype=np.float32,
        )
        cost_vector = np.array(
            [
                total_delay,
                total_energy,
                0.0 if reliability_ok else 1.0,
                1.0 - privacy_score,
            ],
            dtype=np.float32,
        )
        privacy_match = bool(target.security_level >= task.privacy_level)
        violation = bool(
            (not reliability_ok)
            or (task.privacy_level >= self.config.high_privacy_threshold and not privacy_match)
        )

        self._advance_system(
            target_index=action.route_index,
            execution_delay=execution_delay,
            local_energy=local_energy,
        )
        self._enqueue_arrivals(force_non_empty=False)
        if not self.pending_tasks:
            self._enqueue_arrivals(force_non_empty=True)
        self.current_task = self.pending_tasks.popleft()

        info: Dict[str, object] = {
            "task_id": task.task_id,
            "profile": task.profile,
            "privacy_level": task.privacy_level,
            "selected_node": target.node_id,
            "selected_node_index": action.route_index,
            "reward_vector": reward_vector,
            "cost_vector": cost_vector,
            "proxy_reward": float(np.dot(action.proxy_weights, reward_vector)),
            "latency": float(total_delay),
            "energy": float(total_energy),
            "local_energy": float(local_energy),
            "privacy_match": privacy_match,
            "violation": violation,
            "expert_index": action.expert_index,
            "scenario_seed": self.seed,
        }
        return self._build_observation(), info

    def _build_distribution_context(self) -> None:
        self.profile_weights = np.array(
            [profile.weight for profile in self.config.task_profiles],
            dtype=np.float64,
        )
        self.arrival_rates = np.array(self.config.arrival.rates, dtype=np.float64)
        self.arrival_transition = np.array(self.config.arrival.transition, dtype=np.float64)
        self.privacy_ratios = np.array(
            [
                self.config.low_privacy_ratio,
                self.config.medium_privacy_ratio,
                self.config.high_privacy_ratio,
            ],
            dtype=np.float64,
        )

        if self.mode == "unseen":
            shift = self.config.profile_shift_scale
            self.profile_weights = self._normalize(
                self.profile_weights
                + np.array([shift, -0.10, shift * 0.4], dtype=np.float64)
            )
            self.arrival_rates = self.arrival_rates * np.array([1.0, 1.15, 1.40])
            self.arrival_transition = np.array(
                (
                    (0.70, 0.24, 0.06),
                    (0.12, 0.58, 0.30),
                    (0.05, 0.18, 0.77),
                ),
                dtype=np.float64,
            )
            self.privacy_ratios = self._normalize(
                self.privacy_ratios + np.array([-0.10, 0.03, 0.07], dtype=np.float64)
            )

    def _build_local_node(self) -> NodeSpec:
        capacity_shift = 1.0 + (self.config.heterogeneity_shift_scale if self.mode == "unseen" else 0.0)
        return NodeSpec(
            node_id="local_device",
            kind="local",
            compute_capacity=float(self.rng.uniform(55.0, 85.0) / capacity_shift),
            memory_capacity=float(self.rng.uniform(4.0, 8.0)),
            io_rate=float(self.rng.uniform(65.0, 110.0)),
            security_level=self.config.privacy_levels,
            base_power=0.018,
            load_amp=0.55,
            reliability=float(self.rng.uniform(0.93, 0.99)),
            available_bandwidth=float(self.rng.uniform(35.0, 70.0)),
            remaining_energy=1.0,
        )

    def _build_edge_nodes(self) -> List[NodeSpec]:
        nodes: List[NodeSpec] = []
        heterogeneity = self.config.heterogeneity_shift_scale if self.mode == "unseen" else 0.0
        for index in range(self.config.num_edge_nodes):
            nodes.append(
                NodeSpec(
                    node_id=f"edge_{index}",
                    kind="edge",
                    compute_capacity=float(self.rng.uniform(120.0, 220.0) * (1.0 + self.rng.uniform(-heterogeneity, heterogeneity))),
                    memory_capacity=float(self.rng.uniform(12.0, 40.0)),
                    io_rate=float(self.rng.uniform(120.0, 260.0)),
                    security_level=int(self.rng.integers(4, self.config.privacy_levels + 1)),
                    base_power=float(self.rng.uniform(0.05, 0.13)),
                    load_amp=float(self.rng.uniform(0.25, 0.65)),
                    reliability=float(self.rng.uniform(0.82, 0.985)),
                    available_bandwidth=float(self.rng.uniform(45.0, 100.0)),
                )
            )
        return nodes

    def _build_cloud_node(self) -> NodeSpec:
        return NodeSpec(
            node_id="cloud",
            kind="cloud",
            compute_capacity=float(self.rng.uniform(300.0, 480.0)),
            memory_capacity=128.0,
            io_rate=float(self.rng.uniform(260.0, 440.0)),
            security_level=max(5, self.config.high_privacy_threshold - 1),
            base_power=0.16,
            load_amp=0.20,
            reliability=float(self.rng.uniform(0.96, 0.997)),
            available_bandwidth=float(self.rng.uniform(70.0, 125.0)),
        )

    def _enqueue_arrivals(self, force_non_empty: bool) -> None:
        arrivals = 0
        attempts = 0
        while arrivals == 0:
            self._step_mmpp()
            arrivals = int(self.rng.poisson(self.arrival_rates[self.mmpp_state]))
            attempts += 1
            if not force_non_empty or attempts > 3:
                break
        arrivals = max(arrivals, 1 if force_non_empty else 0)
        for _ in range(arrivals):
            self.pending_tasks.append(self._sample_task())

    def _step_mmpp(self) -> None:
        self.mmpp_state = int(
            self.rng.choice(
                np.arange(len(self.arrival_rates)),
                p=self.arrival_transition[self.mmpp_state],
            )
        )

    def _sample_task(self) -> TaskSpec:
        profile = self.rng.choice(self.config.task_profiles, p=self.profile_weights)
        privacy_draw = float(self.rng.random())
        if privacy_draw < self.privacy_ratios[0]:
            privacy_level = int(
                self.rng.integers(1, self.config.medium_privacy_threshold)
            )
        elif privacy_draw < self.privacy_ratios[0] + self.privacy_ratios[1]:
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

        deadline_low, deadline_high = profile.deadline_range
        if self.mode == "unseen":
            deadline_high *= 0.85

        task = TaskSpec(
            task_id=f"task_{self.seed}_{self.task_counter}",
            source_device="local_device",
            profile=profile.name,
            data_size=self._sample_truncated_lognormal(
                profile.data_mu,
                profile.data_sigma * (1.0 + 0.10 * (self.mode == "unseen")),
                *profile.data_bounds,
            ),
            compute_demand=self._sample_truncated_lognormal(
                profile.compute_mu,
                profile.compute_sigma * (1.0 + 0.10 * (self.mode == "unseen")),
                *profile.compute_bounds,
            ),
            memory_demand=float(self.rng.uniform(*profile.memory_range)),
            deadline=float(self.rng.uniform(deadline_low, deadline_high)),
            priority=int(self.rng.integers(profile.priority_range[0], profile.priority_range[1] + 1)),
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
        avg_neighbor_cpu = float(np.mean([node.cpu_utilization for node in neighbors])) if neighbors else 0.0
        avg_neighbor_queue = float(np.mean([node.queue_backlog for node in neighbors])) if neighbors else 0.0
        avg_neighbor_bw = float(np.mean([node.available_bandwidth for node in neighbors])) if neighbors else 0.0
        avg_neighbor_rel = float(np.mean([node.reliability for node in neighbors])) if neighbors else 1.0
        packet_loss = float(
            np.clip(
                self.config.packet_loss_base
                + self.rng.normal(
                    0.0,
                    self.config.rate_noise_std
                    * (1.0 + self.config.link_shift_scale * (self.mode == "unseen"))
                    * 0.08,
                ),
                0.0,
                0.35,
            )
        )
        max_data = max(profile.data_bounds[1] for profile in self.config.task_profiles)
        max_compute = max(profile.compute_bounds[1] for profile in self.config.task_profiles)
        max_deadline = max(profile.deadline_range[1] for profile in self.config.task_profiles)
        medium_or_high = [
            1
            for level in self.recent_privacy_levels
            if level >= self.config.medium_privacy_threshold
        ]
        sensitivity_ratio = float(sum(medium_or_high) / max(len(self.recent_privacy_levels), 1))

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
            dtype=np.float32,
        )
        context = np.array(
            [
                vector[0],
                vector[2],
                vector[1],
                vector[4],
                vector[3],
                vector[5],
                vector[6],
                vector[8],
                1.0 - vector[11],
                sensitivity_ratio,
            ],
            dtype=np.float32,
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
            if (
                self.config.enforce_privacy_mask
                and task.privacy_level >= self.config.high_privacy_threshold
            ):
                privacy_ok = node.security_level >= self.config.high_privacy_threshold or node.kind == "local"
            mask.append(1.0 if memory_ok and privacy_ok else 0.0)
        return np.array(mask, dtype=np.float32)

    def _effective_rate(self, target: NodeSpec, bandwidth_ratio: float) -> float:
        local_rate = self.nodes[0].available_bandwidth
        congestion = np.clip(target.cpu_utilization, 0.0, 1.0)
        noise = float(
            self.rng.normal(
                0.0,
                self.config.rate_noise_std
                * (1.0 + self.config.link_shift_scale * (self.mode == "unseen")),
            )
        )
        rate = (
            local_rate
            * bandwidth_ratio
            * (1.0 - self.config.congestion_sensitivity * congestion)
            * (1.0 + noise)
        )
        return float(np.clip(rate, self.config.min_uplink_rate, self.config.max_uplink_rate))

    def _compute_energy(
        self,
        task: TaskSpec,
        target: NodeSpec,
        execution_delay: float,
        transmission_delay: float,
        cpu_ratio: float,
        tx_power: float,
    ) -> Tuple[float, float]:
        if target.kind == "local":
            freq = self.config.local_cpu_frequency * cpu_ratio
            local_energy = float(self.config.local_dvfs_kappa * (freq ** 2) * task.compute_demand)
            return local_energy, local_energy

        comm_energy = tx_power * transmission_delay
        remote_energy = float(
            target.base_power * execution_delay * (1.0 + target.load_amp * target.cpu_utilization)
        )
        return float(comm_energy), float(comm_energy + remote_energy)

    def _privacy_score(self, task: TaskSpec, target: NodeSpec) -> float:
        gap = max(0, task.privacy_level - target.security_level)
        return float(1.0 - gap / max(self.config.privacy_levels - 1, 1))

    def _advance_system(
        self, target_index: int, execution_delay: float, local_energy: float
    ) -> None:
        self.current_time += 1
        for index, node in enumerate(self.nodes):
            node.queue_backlog = max(0.0, node.queue_backlog - self.config.queue_decay_per_step)
            if index == target_index:
                node.queue_backlog += execution_delay
            node.cpu_utilization = float(np.clip(node.queue_backlog / self.config.queue_scale, 0.0, 1.0))
            node.available_bandwidth = float(
                np.clip(
                    node.available_bandwidth + self.rng.normal(0.0, 4.0),
                    self.config.min_uplink_rate,
                    self.config.max_uplink_rate + 35.0,
                )
            )
            node.reliability = float(
                np.clip(
                    node.reliability - 0.012 * node.cpu_utilization + self.rng.normal(0.0, 0.004),
                    0.75,
                    0.997,
                )
            )
        self.nodes[0].remaining_energy = float(
            np.clip(self.nodes[0].remaining_energy - local_energy / (self.config.energy_budget_max * 20.0), 0.0, 1.0)
        )

    @staticmethod
    def _normalize(values: np.ndarray) -> np.ndarray:
        clipped = np.clip(values, 1e-6, None)
        return clipped / np.sum(clipped)
