from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

from src.data.schemas import ActionRecord, Observation


@dataclass(frozen=True)
class AdapterProfile:
    method: str
    objective_preference: tuple[float, float, float, float]
    exploration_scale: float
    federated: bool = False


class PaperBaselineAdapter:
    """Runnable smoke adapter for paper-level baselines under the shared action interface."""

    def __init__(self, profile: AdapterProfile, seed: int = 0):
        self.profile = profile
        self.method = profile.method
        self.objective_preference = np.asarray(profile.objective_preference, dtype=np.float32)
        self.objective_preference = self.objective_preference / np.sum(self.objective_preference)
        self.rng = np.random.default_rng(seed)

    def select_action(self, observation: Observation) -> ActionRecord:
        feasible = np.flatnonzero(observation.feasible_mask > 0)
        if feasible.size == 0:
            raise ValueError("Observation has no feasible route.")

        route_scores = np.array(
            [self._route_score(observation, index) for index in range(len(observation.candidate_nodes))],
            dtype=np.float32,
        )
        route_scores[observation.feasible_mask <= 0] = -np.inf
        route_index = int(np.argmax(route_scores))

        route_onehot = np.zeros_like(observation.feasible_mask, dtype=np.float32)
        route_onehot[route_index] = 1.0
        continuous_action = self._continuous_action(observation)
        expert_index = int(np.argmax(self.objective_preference))
        expert_probs = self._expert_probs(expert_index)

        return ActionRecord(
            route_index=route_index,
            route_onehot=route_onehot,
            continuous_action=continuous_action,
            expert_index=expert_index,
            expert_probs=expert_probs,
            proxy_weights=self.objective_preference.astype(np.float32),
            sequence_obs=np.zeros((1, observation.vector.shape[0]), dtype=np.float32),
        )

    def _route_score(self, observation: Observation, route_index: int) -> float:
        node_name = observation.candidate_nodes[route_index]
        latency_w, energy_w, reliability_w, privacy_w = self.objective_preference
        urgency = float(observation.context[8])
        privacy_level = float(observation.vector[13])
        compute_pressure = float(observation.vector[10])
        local_energy = float(observation.vector[3])
        neighbor_rel = float(observation.vector[8])
        neighbor_queue = float(observation.vector[6])

        if node_name == "local_device":
            latency_score = 0.48 - 0.20 * compute_pressure - 0.12 * urgency
            energy_score = 0.86 if local_energy > 0.25 else 0.54
            reliability_score = 0.74
            privacy_score = 0.98
        elif node_name == "cloud":
            latency_score = 0.66 + 0.16 * compute_pressure - 0.10 * urgency
            energy_score = 0.44
            reliability_score = 0.84
            privacy_score = 0.48 - 0.24 * privacy_level
        else:
            latency_score = 0.82 - 0.24 * neighbor_queue + 0.08 * urgency
            energy_score = 0.62
            reliability_score = 0.52 + 0.44 * neighbor_rel
            privacy_score = 0.66 - 0.12 * privacy_level

        if self.profile.federated:
            reliability_score += 0.04
            privacy_score += 0.06

        return float(
            latency_w * latency_score
            + energy_w * energy_score
            + reliability_w * reliability_score
            + privacy_w * privacy_score
        )

    def _continuous_action(self, observation: Observation) -> np.ndarray:
        latency_w, energy_w, reliability_w, privacy_w = self.objective_preference
        urgency = float(observation.context[8])
        sensitivity = float(observation.context[9])
        noise = self.rng.normal(0.0, self.profile.exploration_scale, size=3)
        raw = np.array(
            [
                0.32 + 0.50 * latency_w + 0.18 * reliability_w - 0.18 * energy_w,
                0.38 + 0.42 * latency_w + 0.24 * reliability_w + 0.08 * urgency,
                0.12 + 0.62 * latency_w - 0.34 * energy_w - 0.08 * privacy_w,
            ],
            dtype=np.float32,
        )
        raw[0] += 0.08 * urgency
        raw[2] -= 0.06 * sensitivity * privacy_w
        return np.clip(raw + noise, [0.05, 0.05, 0.0], [1.0, 1.0, 1.0]).astype(np.float32)

    def _expert_probs(self, expert_index: int) -> np.ndarray:
        probs = np.full(4, 0.10, dtype=np.float32)
        probs[expert_index] = 0.70
        return probs / np.sum(probs)


class TrainableActorCriticBaseline(PaperBaselineAdapter):
    critic_count = 1

    def __init__(
        self,
        profile: AdapterProfile,
        *,
        seed: int = 0,
        batch_size: int = 32,
        gamma: float = 0.99,
        tau: float = 0.02,
        actor_lr: float = 1e-3,
        critic_lr: float = 1e-3,
        alpha: float = 0.05,
    ):
        super().__init__(profile, seed=seed)
        self.batch_size = batch_size
        self.gamma = gamma
        self.tau = tau
        self.alpha = alpha
        self.training_steps = 0
        self.replay: list[dict[str, np.ndarray | float]] = []
        self.max_replay_size = 5000
        tf.random.set_seed(seed)

        self.actor = self._build_actor(f"{self.method.lower().replace('-', '_')}_actor")
        self.target_actor = keras.models.clone_model(self.actor)
        self.target_actor.set_weights(self.actor.get_weights())
        self.critics = [
            self._build_critic(f"{self.method.lower().replace('-', '_')}_critic_{idx}")
            for idx in range(self.critic_count)
        ]
        self.target_critics = [keras.models.clone_model(critic) for critic in self.critics]
        for target, source in zip(self.target_critics, self.critics):
            target.set_weights(source.get_weights())
        self.actor_optimizer = keras.optimizers.Adam(actor_lr)
        self.critic_optimizers = [keras.optimizers.Adam(critic_lr) for _ in self.critics]

    def select_action(self, observation: Observation) -> ActionRecord:
        obs_batch = tf.convert_to_tensor(observation.vector[None, :], dtype=tf.float32)
        ctx_batch = tf.convert_to_tensor(observation.context[None, :], dtype=tf.float32)
        route_logits, continuous_logits = self.actor([obs_batch, ctx_batch], training=False)
        mask_batch = tf.convert_to_tensor(observation.feasible_mask[None, :], dtype=tf.float32)
        masked_logits = tf.where(mask_batch > 0, route_logits, tf.constant(-1.0e9, dtype=tf.float32))
        route_index = int(tf.argmax(masked_logits[0]).numpy())

        route_onehot = np.zeros_like(observation.feasible_mask, dtype=np.float32)
        route_onehot[route_index] = 1.0
        continuous = tf.sigmoid(continuous_logits)[0].numpy().astype(np.float32)
        if self.profile.exploration_scale > 0:
            continuous = np.clip(
                continuous + self.rng.normal(0.0, self.profile.exploration_scale, size=3),
                0.0,
                1.0,
            ).astype(np.float32)

        expert_index = int(np.argmax(self.objective_preference))
        return ActionRecord(
            route_index=route_index,
            route_onehot=route_onehot,
            continuous_action=continuous,
            expert_index=expert_index,
            expert_probs=self._expert_probs(expert_index),
            proxy_weights=self.objective_preference.astype(np.float32),
            sequence_obs=np.zeros((1, observation.vector.shape[0]), dtype=np.float32),
        )

    def observe_transition(
        self,
        observation: Observation,
        action: ActionRecord,
        reward_vector: np.ndarray,
        next_observation: Observation,
        done: bool,
    ) -> None:
        self.replay.append(
            {
                "obs": np.asarray(observation.vector, dtype=np.float32),
                "context": np.asarray(observation.context, dtype=np.float32),
                "mask": np.asarray(observation.feasible_mask, dtype=np.float32),
                "action_route": np.asarray(action.route_onehot, dtype=np.float32),
                "action_continuous": np.asarray(action.continuous_action, dtype=np.float32),
                "reward": float(np.dot(self.objective_preference, reward_vector)),
                "next_obs": np.asarray(next_observation.vector, dtype=np.float32),
                "next_context": np.asarray(next_observation.context, dtype=np.float32),
                "next_mask": np.asarray(next_observation.feasible_mask, dtype=np.float32),
                "done": float(done),
            }
        )
        if len(self.replay) > self.max_replay_size:
            self.replay.pop(0)

    def train_step(self) -> dict[str, float]:
        if len(self.replay) < self.batch_size:
            return {"actor_loss": 0.0, "critic_loss": 0.0, "entropy_loss": 0.0}

        batch = self._sample_batch()
        target_route, target_continuous, entropy = self._target_policy(
            batch["next_obs"],
            batch["next_context"],
            batch["next_mask"],
        )
        target_q_values = [
            target_critic(
                [
                    batch["next_obs"],
                    batch["next_context"],
                    target_route,
                    target_continuous,
                ],
                training=False,
            )
            for target_critic in self.target_critics
        ]
        target_q = self._target_q(target_q_values, entropy)
        y = batch["reward"] + self.gamma * (1.0 - batch["done"]) * target_q

        critic_losses = []
        for critic, optimizer in zip(self.critics, self.critic_optimizers):
            with tf.GradientTape() as tape:
                q = critic(
                    [
                        batch["obs"],
                        batch["context"],
                        batch["action_route"],
                        batch["action_continuous"],
                    ],
                    training=True,
                )
                loss = tf.reduce_mean(tf.square(tf.stop_gradient(y) - q))
            grads = tape.gradient(loss, critic.trainable_variables)
            optimizer.apply_gradients(zip(grads, critic.trainable_variables))
            critic_losses.append(float(loss.numpy()))

        with tf.GradientTape() as tape:
            route_probs, continuous, actor_entropy = self._policy(
                batch["obs"],
                batch["context"],
                batch["mask"],
                actor=self.actor,
                training=True,
            )
            q_values = [
                critic([batch["obs"], batch["context"], route_probs, continuous], training=False)
                for critic in self.critics
            ]
            actor_loss = self._actor_loss(q_values, actor_entropy)
        actor_grads = tape.gradient(actor_loss, self.actor.trainable_variables)
        self.actor_optimizer.apply_gradients(zip(actor_grads, self.actor.trainable_variables))

        self._soft_update_targets()
        self.training_steps += 1
        return {
            "actor_loss": float(actor_loss.numpy()),
            "critic_loss": float(np.mean(critic_losses)),
            "entropy_loss": float(-tf.reduce_mean(actor_entropy).numpy()),
        }

    def _build_actor(self, name: str) -> keras.Model:
        obs_input = keras.Input(shape=(14,))
        context_input = keras.Input(shape=(10,))
        x = layers.Concatenate()([obs_input, context_input])
        x = layers.Dense(64, activation="relu")(x)
        x = layers.Dense(64, activation="relu")(x)
        route_logits = layers.Dense(8)(x)
        continuous_logits = layers.Dense(3)(x)
        return keras.Model([obs_input, context_input], [route_logits, continuous_logits], name=name)

    def _build_critic(self, name: str) -> keras.Model:
        obs_input = keras.Input(shape=(14,))
        context_input = keras.Input(shape=(10,))
        route_input = keras.Input(shape=(8,))
        continuous_input = keras.Input(shape=(3,))
        x = layers.Concatenate()([obs_input, context_input, route_input, continuous_input])
        x = layers.Dense(96, activation="relu")(x)
        x = layers.Dense(96, activation="relu")(x)
        q_value = layers.Dense(1)(x)
        return keras.Model(
            [obs_input, context_input, route_input, continuous_input],
            q_value,
            name=name,
        )

    def _sample_batch(self) -> dict[str, tf.Tensor]:
        indices = self.rng.choice(len(self.replay), size=self.batch_size, replace=False)
        rows = [self.replay[int(index)] for index in indices]
        return {
            "obs": tf.convert_to_tensor(np.stack([row["obs"] for row in rows]), dtype=tf.float32),
            "context": tf.convert_to_tensor(np.stack([row["context"] for row in rows]), dtype=tf.float32),
            "mask": tf.convert_to_tensor(np.stack([row["mask"] for row in rows]), dtype=tf.float32),
            "action_route": tf.convert_to_tensor(np.stack([row["action_route"] for row in rows]), dtype=tf.float32),
            "action_continuous": tf.convert_to_tensor(np.stack([row["action_continuous"] for row in rows]), dtype=tf.float32),
            "reward": tf.convert_to_tensor(np.array([row["reward"] for row in rows], dtype=np.float32)[:, None]),
            "next_obs": tf.convert_to_tensor(np.stack([row["next_obs"] for row in rows]), dtype=tf.float32),
            "next_context": tf.convert_to_tensor(np.stack([row["next_context"] for row in rows]), dtype=tf.float32),
            "next_mask": tf.convert_to_tensor(np.stack([row["next_mask"] for row in rows]), dtype=tf.float32),
            "done": tf.convert_to_tensor(np.array([row["done"] for row in rows], dtype=np.float32)[:, None]),
        }

    def _target_policy(
        self,
        obs: tf.Tensor,
        context: tf.Tensor,
        mask: tf.Tensor,
    ) -> tuple[tf.Tensor, tf.Tensor, tf.Tensor]:
        return self._policy(obs, context, mask, actor=self.target_actor, training=False)

    def _policy(
        self,
        obs: tf.Tensor,
        context: tf.Tensor,
        mask: tf.Tensor,
        *,
        actor: keras.Model,
        training: bool,
    ) -> tuple[tf.Tensor, tf.Tensor, tf.Tensor]:
        route_logits, continuous_logits = actor([obs, context], training=training)
        masked_logits = tf.where(mask > 0, route_logits, tf.constant(-1.0e9, dtype=tf.float32))
        route_probs = tf.nn.softmax(masked_logits, axis=1)
        continuous = tf.sigmoid(continuous_logits)
        entropy = -tf.reduce_sum(route_probs * tf.math.log(route_probs + 1e-8), axis=1, keepdims=True)
        return route_probs, continuous, entropy

    def _target_q(self, target_q_values: list[tf.Tensor], entropy: tf.Tensor) -> tf.Tensor:
        return target_q_values[0]

    def _actor_loss(self, q_values: list[tf.Tensor], entropy: tf.Tensor) -> tf.Tensor:
        return -tf.reduce_mean(q_values[0])

    def _soft_update_targets(self) -> None:
        self._soft_update(self.target_actor, self.actor)
        for target, source in zip(self.target_critics, self.critics):
            self._soft_update(target, source)

    def _soft_update(self, target: keras.Model, source: keras.Model) -> None:
        updated = []
        for target_weight, source_weight in zip(target.get_weights(), source.get_weights()):
            updated.append((1.0 - self.tau) * target_weight + self.tau * source_weight)
        target.set_weights(updated)


class TrainableSacBaseline(TrainableActorCriticBaseline):
    critic_count = 2

    def _target_q(self, target_q_values: list[tf.Tensor], entropy: tf.Tensor) -> tf.Tensor:
        return tf.minimum(target_q_values[0], target_q_values[1]) + self.alpha * entropy

    def _actor_loss(self, q_values: list[tf.Tensor], entropy: tf.Tensor) -> tf.Tensor:
        min_q = tf.minimum(q_values[0], q_values[1])
        return tf.reduce_mean(-min_q - self.alpha * entropy)


class DdpgAdapter(PaperBaselineAdapter):
    def __init__(self, seed: int = 0):
        super().__init__(
            AdapterProfile("DDPG", (0.55, 0.15, 0.20, 0.10), 0.00),
            seed=seed,
        )


class SacAdapter(PaperBaselineAdapter):
    def __init__(self, seed: int = 0):
        super().__init__(
            AdapterProfile("SAC", (0.25, 0.25, 0.25, 0.25), 0.04),
            seed=seed,
        )


class MoDdpgAdapter(TrainableActorCriticBaseline):
    def __init__(self, seed: int = 0, batch_size: int = 32):
        super().__init__(
            AdapterProfile("MO-DDPG", (0.42, 0.20, 0.24, 0.14), 0.00),
            seed=seed,
            batch_size=batch_size,
        )


class PmSoftMoeAdapter(PaperBaselineAdapter):
    def __init__(self, seed: int = 0):
        super().__init__(
            AdapterProfile("PM-Soft-MoE", (0.25, 0.25, 0.25, 0.25), 0.01),
            seed=seed,
        )


class MoSacAdapter(TrainableSacBaseline):
    def __init__(self, seed: int = 0, batch_size: int = 32):
        super().__init__(
            AdapterProfile("MO-SAC", (0.25, 0.25, 0.25, 0.25), 0.04),
            seed=seed,
            batch_size=batch_size,
        )


class CapqlAdapter(PaperBaselineAdapter):
    def __init__(self, seed: int = 0):
        super().__init__(
            AdapterProfile("CAPQL", (0.20, 0.18, 0.24, 0.38), 0.02),
            seed=seed,
        )


class PgmorlAdapter(PaperBaselineAdapter):
    def __init__(self, seed: int = 0):
        super().__init__(
            AdapterProfile("PGMORL", (0.22, 0.20, 0.36, 0.22), 0.02),
            seed=seed,
        )


class FlMoSacAdapter(PaperBaselineAdapter):
    def __init__(self, seed: int = 0):
        super().__init__(
            AdapterProfile("FL-MO-SAC", (0.23, 0.23, 0.28, 0.26), 0.03, federated=True),
            seed=seed,
        )


class FedMorlAdapter(PaperBaselineAdapter):
    def __init__(self, seed: int = 0):
        super().__init__(
            AdapterProfile("FedMORL", (0.18, 0.20, 0.28, 0.34), 0.02, federated=True),
            seed=seed,
        )


def available_core_baseline_adapters(seed: int = 0) -> dict[str, PaperBaselineAdapter]:
    return {
        "PM-Soft-MoE": PmSoftMoeAdapter(seed=seed),
        "MO-DDPG": MoDdpgAdapter(seed=seed + 1),
        "MO-SAC": MoSacAdapter(seed=seed + 2),
        "CAPQL": CapqlAdapter(seed=seed + 3),
        "PGMORL": PgmorlAdapter(seed=seed + 4),
        "FL-MO-SAC": FlMoSacAdapter(seed=seed + 5),
        "FedMORL": FedMorlAdapter(seed=seed + 6),
    }
