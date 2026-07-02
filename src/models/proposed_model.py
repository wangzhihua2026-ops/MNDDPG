from __future__ import annotations

from collections import deque
from typing import Dict, List, Sequence, Tuple

import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

from src.data.schemas import ActionRecord, Observation
from src.models.replay import PrioritizedReplayBuffer, Transition
from src.utils.paper_config import AgentConfig


class PaperAlignedMNDDPGAgent:
    """MNDDPG implementation aligned with the algorithm section of main.tex."""

    def __init__(self, config: AgentConfig, seed: int = 0):
        self.config = config
        self.seed = seed
        np.random.seed(seed)
        tf.random.set_seed(seed)

        self.replay_buffer = PrioritizedReplayBuffer(
            capacity=config.replay_capacity,
            alpha=config.per_alpha,
            beta_start=config.per_beta_start,
            beta_increment=config.per_beta_increment,
        )
        self.obs_history: deque[np.ndarray] = deque(maxlen=config.sequence_length)

        self.shared_encoder = self._build_shared_encoder()
        self.target_shared_encoder = keras.models.clone_model(self.shared_encoder)
        self.target_shared_encoder.set_weights(self.shared_encoder.get_weights())

        self.gating_network = self._build_gating_network()
        self.target_gating_network = keras.models.clone_model(self.gating_network)
        self.target_gating_network.set_weights(self.gating_network.get_weights())

        self.proxy_network = self._build_proxy_network()

        self.expert_actors = self._build_expert_actors()
        self.target_expert_actors = [keras.models.clone_model(actor) for actor in self.expert_actors]
        for target, source in zip(self.target_expert_actors, self.expert_actors):
            target.set_weights(source.get_weights())

        self.critics = [self._build_critic(f"critic_{idx}") for idx in range(config.num_objectives)]
        self.target_critics = [keras.models.clone_model(critic) for critic in self.critics]
        for target, source in zip(self.target_critics, self.critics):
            target.set_weights(source.get_weights())

        self.actor_optimizer = keras.optimizers.Adam(config.actor_lr)
        self.critic_optimizers = [
            keras.optimizers.Adam(config.critic_lr) for _ in range(config.num_objectives)
        ]
        self.proxy_optimizer = keras.optimizers.Adam(config.proxy_lr)

        self.model_groups = {
            "shared": self.shared_encoder,
            "gating": self.gating_network,
            "proxy": self.proxy_network,
            **{f"actor_{idx}": model for idx, model in enumerate(self.expert_actors)},
            **{f"critic_{idx}": model for idx, model in enumerate(self.critics)},
        }

    def reset_episode(self) -> None:
        self.obs_history.clear()

    def preview_sequence(self, obs_vector: np.ndarray) -> np.ndarray:
        history = list(self.obs_history) + [np.asarray(obs_vector, dtype=np.float32)]
        history = history[-self.config.sequence_length :]
        padded = [np.zeros(self.config.observation_dim, dtype=np.float32)] * (
            self.config.sequence_length - len(history)
        ) + history
        return np.stack(padded, axis=0).astype(np.float32)

    def commit_observation(self, obs_vector: np.ndarray) -> None:
        self.obs_history.append(np.asarray(obs_vector, dtype=np.float32))

    def select_action(self, observation: Observation, training: bool = True) -> ActionRecord:
        sequence_obs = self.preview_sequence(observation.vector)
        obs_batch = tf.convert_to_tensor(observation.vector[None, :], dtype=tf.float32)
        ctx_batch = tf.convert_to_tensor(observation.context[None, :], dtype=tf.float32)
        seq_batch = tf.convert_to_tensor(sequence_obs[None, :, :], dtype=tf.float32)
        mask_batch = tf.convert_to_tensor(observation.feasible_mask[None, :], dtype=tf.float32)

        features = self.shared_encoder(obs_batch, training=False)
        all_route_logits, all_continuous = self._all_expert_outputs(
            features, ctx_batch, seq_batch, use_target=False, training=False
        )
        gating_logits = self.gating_network(
            tf.concat([features, ctx_batch], axis=1), training=False
        )
        gating_probs = tf.nn.softmax(gating_logits, axis=1)
        if not self.config.use_context_gating:
            expert_index = 0
            expert_probs = np.eye(self.config.num_experts, dtype=np.float32)[expert_index]
            gating_selector = tf.one_hot([expert_index], depth=self.config.num_experts)
        elif training:
            gating_selector = self._straight_through_categorical(
                gating_logits, self.config.gating_temperature
            )
            expert_index = int(tf.argmax(gating_selector[0]).numpy())
            expert_probs = gating_probs[0].numpy()
        else:
            expert_index = int(tf.argmax(gating_probs[0]).numpy())
            expert_probs = gating_probs[0].numpy()
            gating_selector = tf.one_hot([expert_index], depth=self.config.num_experts)

        if self.config.use_hard_routing:
            selected_route_logits = all_route_logits[:, expert_index, :]
            selected_continuous = all_continuous[:, expert_index, :]
        else:
            selected_route_logits = tf.reduce_sum(
                all_route_logits * gating_selector[:, :, None], axis=1
            )
            selected_continuous = tf.reduce_sum(
                all_continuous * gating_selector[:, :, None], axis=1
            )
        masked_logits = tf.where(mask_batch > 0, selected_route_logits, tf.constant(-1.0e9, dtype=tf.float32))
        route_onehot = tf.one_hot(tf.argmax(masked_logits, axis=1), depth=self.config.num_routes)
        continuous = self._project_continuous(selected_continuous)
        proxy_weights = self._proxy_weights(ctx_batch, training=False)[0].numpy()

        return ActionRecord(
            route_index=int(tf.argmax(route_onehot[0]).numpy()),
            route_onehot=route_onehot[0].numpy().astype(np.float32),
            continuous_action=continuous[0].numpy().astype(np.float32),
            expert_index=expert_index,
            expert_probs=expert_probs.astype(np.float32),
            proxy_weights=proxy_weights.astype(np.float32),
            sequence_obs=sequence_obs,
        )

    def store_transition(
        self,
        observation: Observation,
        action: ActionRecord,
        reward_vector: np.ndarray,
        next_observation: Observation,
        next_sequence_obs: np.ndarray,
        done: bool,
    ) -> None:
        transition = Transition(
            obs=np.asarray(observation.vector, dtype=np.float32),
            context=np.asarray(observation.context, dtype=np.float32),
            mask=np.asarray(observation.feasible_mask, dtype=np.float32),
            seq_obs=np.asarray(action.sequence_obs, dtype=np.float32),
            action_route=np.asarray(action.route_onehot, dtype=np.float32),
            action_continuous=np.asarray(action.continuous_action, dtype=np.float32),
            reward_vector=np.asarray(reward_vector, dtype=np.float32),
            next_obs=np.asarray(next_observation.vector, dtype=np.float32),
            next_context=np.asarray(next_observation.context, dtype=np.float32),
            next_mask=np.asarray(next_observation.feasible_mask, dtype=np.float32),
            next_seq_obs=np.asarray(next_sequence_obs, dtype=np.float32),
            done=float(done),
        )
        self.replay_buffer.add(transition)

    def train_step(self) -> Dict[str, float]:
        if len(self.replay_buffer) < self.config.batch_size:
            return {
                "critic_loss": 0.0,
                "actor_loss": 0.0,
                "proxy_loss": 0.0,
            }

        batch, indices, is_weights = self.replay_buffer.sample(self.config.batch_size)
        tensors = self._batch_to_tensors(batch, is_weights)

        target_action = self._target_action(
            tensors["next_obs"],
            tensors["next_context"],
            tensors["next_mask"],
            tensors["next_seq_obs"],
        )
        target_q_values = [
            self.target_critics[idx](
                [
                    self.target_shared_encoder(tensors["next_obs"], training=False),
                    target_action["route_onehot"],
                    target_action["continuous"],
                    tensors["next_context"],
                ],
                training=False,
            )
            for idx in range(self.config.num_objectives)
        ]

        td_errors_per_objective: List[tf.Tensor] = []
        critic_losses: List[float] = []

        for objective_index in range(self.config.num_objectives):
            reward = tensors["reward_vector"][:, objective_index : objective_index + 1]
            y = reward + self.config.gamma * target_q_values[objective_index] * (1.0 - tensors["done"])

            with tf.GradientTape() as tape:
                current_features = self.shared_encoder(tensors["obs"], training=True)
                q = self.critics[objective_index](
                    [
                        current_features,
                        tensors["action_route"],
                        tensors["action_continuous"],
                        tensors["context"],
                    ],
                    training=True,
                )
                td_error = y - q
                td_errors_per_objective.append(td_error)
                loss = tf.reduce_mean(tensors["is_weights"] * tf.square(td_error))
            variables = (
                self.critics[objective_index].trainable_variables
                + self.shared_encoder.trainable_variables
            )
            grads = tape.gradient(loss, variables)
            self.critic_optimizers[objective_index].apply_gradients(zip(grads, variables))
            critic_losses.append(float(loss.numpy()))

        with tf.GradientTape() as tape:
            features = self.shared_encoder(tensors["obs"], training=True)
            action = self._policy_action(
                tensors["obs"],
                tensors["context"],
                tensors["mask"],
                tensors["seq_obs"],
                features=features,
                training=True,
            )
            q_values = []
            for objective_index in range(self.config.num_objectives):
                q_values.append(
                    self.critics[objective_index](
                        [features, action["route_onehot"], action["continuous"], tensors["context"]],
                        training=False,
                    )
                )
            q_stack = tf.concat(q_values, axis=1)
            actor_loss = -tf.reduce_mean(tf.reduce_sum(action["proxy_weights"] * q_stack, axis=1))
        actor_variables = (
            self.shared_encoder.trainable_variables
            + self.gating_network.trainable_variables
            + sum((actor.trainable_variables for actor in self.expert_actors), [])
        )
        actor_grads = tape.gradient(actor_loss, actor_variables)
        self.actor_optimizer.apply_gradients(zip(actor_grads, actor_variables))

        with tf.GradientTape() as tape:
            features = self.shared_encoder(tensors["obs"], training=False)
            action = self._policy_action(
                tensors["obs"],
                tensors["context"],
                tensors["mask"],
                tensors["seq_obs"],
                features=features,
                training=False,
            )
            q_values = []
            for objective_index in range(self.config.num_objectives):
                q_values.append(
                    self.critics[objective_index](
                        [features, action["route_onehot"], action["continuous"], tensors["context"]],
                        training=False,
                    )
                )
            q_stack = tf.concat(q_values, axis=1)
            alpha = tf.nn.softmax(self.proxy_network(tensors["context"], training=True), axis=1)
            entropy = -tf.reduce_sum(alpha * tf.math.log(alpha + 1e-8), axis=1)
            proxy_objective = tf.reduce_sum(alpha * tf.stop_gradient(q_stack), axis=1)
            proxy_loss = -tf.reduce_mean(proxy_objective + self.config.entropy_lambda * entropy)
        proxy_grads = tape.gradient(proxy_loss, self.proxy_network.trainable_variables)
        self.proxy_optimizer.apply_gradients(zip(proxy_grads, self.proxy_network.trainable_variables))

        td_errors = tf.reduce_mean(
            tf.concat([tf.abs(error) for error in td_errors_per_objective], axis=1), axis=1
        ).numpy()
        self.replay_buffer.update_priorities(indices, td_errors)
        self._soft_update_targets()

        return {
            "critic_loss": float(np.mean(critic_losses)),
            "actor_loss": float(actor_loss.numpy()),
            "proxy_loss": float(proxy_loss.numpy()),
        }

    def get_federated_weights(self) -> Dict[str, np.ndarray]:
        flattened: Dict[str, np.ndarray] = {}
        for group_name, model in self.model_groups.items():
            for index, weight in enumerate(model.get_weights()):
                flattened[f"{group_name}/{index}"] = np.asarray(weight, dtype=np.float32)
        return flattened

    def set_federated_weights(self, weights: Dict[str, np.ndarray]) -> None:
        grouped: Dict[str, List[Tuple[int, np.ndarray]]] = {}
        for key, value in weights.items():
            group_name, index_str = key.split("/")
            grouped.setdefault(group_name, []).append((int(index_str), value))
        for group_name, model in self.model_groups.items():
            if group_name not in grouped:
                continue
            ordered = [value for _, value in sorted(grouped[group_name], key=lambda item: item[0])]
            model.set_weights(ordered)
        self._hard_sync_targets()

    def _hard_sync_targets(self) -> None:
        self.target_shared_encoder.set_weights(self.shared_encoder.get_weights())
        self.target_gating_network.set_weights(self.gating_network.get_weights())
        for target, source in zip(self.target_expert_actors, self.expert_actors):
            target.set_weights(source.get_weights())
        for target, source in zip(self.target_critics, self.critics):
            target.set_weights(source.get_weights())

    def _soft_update_targets(self) -> None:
        self._soft_update_model(self.target_shared_encoder, self.shared_encoder)
        self._soft_update_model(self.target_gating_network, self.gating_network)
        for target, source in zip(self.target_expert_actors, self.expert_actors):
            self._soft_update_model(target, source)
        for target, source in zip(self.target_critics, self.critics):
            self._soft_update_model(target, source)

    def _soft_update_model(self, target: keras.Model, source: keras.Model) -> None:
        updated = []
        for target_weight, source_weight in zip(target.get_weights(), source.get_weights()):
            updated.append(
                (1.0 - self.config.tau) * target_weight + self.config.tau * source_weight
            )
        target.set_weights(updated)

    def _batch_to_tensors(
        self, batch: Sequence[Transition], is_weights: np.ndarray
    ) -> Dict[str, tf.Tensor]:
        return {
            "obs": tf.convert_to_tensor(np.stack([item.obs for item in batch]), dtype=tf.float32),
            "context": tf.convert_to_tensor(np.stack([item.context for item in batch]), dtype=tf.float32),
            "mask": tf.convert_to_tensor(np.stack([item.mask for item in batch]), dtype=tf.float32),
            "seq_obs": tf.convert_to_tensor(np.stack([item.seq_obs for item in batch]), dtype=tf.float32),
            "action_route": tf.convert_to_tensor(np.stack([item.action_route for item in batch]), dtype=tf.float32),
            "action_continuous": tf.convert_to_tensor(np.stack([item.action_continuous for item in batch]), dtype=tf.float32),
            "reward_vector": tf.convert_to_tensor(np.stack([item.reward_vector for item in batch]), dtype=tf.float32),
            "next_obs": tf.convert_to_tensor(np.stack([item.next_obs for item in batch]), dtype=tf.float32),
            "next_context": tf.convert_to_tensor(np.stack([item.next_context for item in batch]), dtype=tf.float32),
            "next_mask": tf.convert_to_tensor(np.stack([item.next_mask for item in batch]), dtype=tf.float32),
            "next_seq_obs": tf.convert_to_tensor(np.stack([item.next_seq_obs for item in batch]), dtype=tf.float32),
            "done": tf.convert_to_tensor(np.array([item.done for item in batch], dtype=np.float32)[:, None]),
            "is_weights": tf.convert_to_tensor(is_weights[:, None], dtype=tf.float32),
        }

    def _target_action(
        self,
        next_obs: tf.Tensor,
        next_context: tf.Tensor,
        next_mask: tf.Tensor,
        next_seq_obs: tf.Tensor,
    ) -> Dict[str, tf.Tensor]:
        next_features = self.target_shared_encoder(next_obs, training=False)
        all_route_logits, all_continuous = self._all_expert_outputs(
            next_features, next_context, next_seq_obs, use_target=True, training=False
        )
        gating_logits = self.target_gating_network(
            tf.concat([next_features, next_context], axis=1),
            training=False,
        )
        gating_hard = self._gating_selector(gating_logits, training=False)
        selected_route_logits = tf.reduce_sum(
            all_route_logits * gating_hard[:, :, None], axis=1
        )
        masked_logits = tf.where(next_mask > 0, selected_route_logits, tf.constant(-1.0e9, dtype=tf.float32))
        route_onehot = tf.one_hot(
            tf.argmax(masked_logits, axis=1), depth=self.config.num_routes
        )
        selected_continuous = tf.reduce_sum(
            all_continuous * gating_hard[:, :, None], axis=1
        )
        continuous = self._project_continuous(selected_continuous)
        return {
            "route_onehot": route_onehot,
            "continuous": continuous,
        }

    def _policy_action(
        self,
        obs: tf.Tensor,
        context: tf.Tensor,
        mask: tf.Tensor,
        seq_obs: tf.Tensor,
        *,
        features: tf.Tensor | None = None,
        training: bool,
    ) -> Dict[str, tf.Tensor]:
        features = self.shared_encoder(obs, training=training) if features is None else features
        all_route_logits, all_continuous = self._all_expert_outputs(
            features, context, seq_obs, use_target=False, training=training
        )
        joint = tf.concat([features, context], axis=1)
        gating_logits = self.gating_network(joint, training=training)
        gating_selector = self._gating_selector(gating_logits, training=training)
        selected_route_logits = tf.reduce_sum(
            all_route_logits * gating_selector[:, :, None], axis=1
        )
        masked_logits = tf.where(mask > 0, selected_route_logits, tf.constant(-1.0e9, dtype=tf.float32))
        if self.config.use_hard_routing:
            route_onehot = self._straight_through_categorical(
                masked_logits, self.config.route_temperature
            )
        else:
            route_onehot = tf.nn.softmax(masked_logits / max(self.config.route_temperature, 1e-6), axis=1)
        selected_continuous = tf.reduce_sum(
            all_continuous * gating_selector[:, :, None], axis=1
        )
        continuous = self._project_continuous(selected_continuous)
        proxy_weights = self._proxy_weights(context, training=training)
        return {
            "route_onehot": route_onehot,
            "continuous": continuous,
            "proxy_weights": proxy_weights,
        }

    def _gating_selector(self, logits: tf.Tensor, *, training: bool) -> tf.Tensor:
        batch_size = tf.shape(logits)[0]
        if not self.config.use_context_gating:
            return tf.one_hot(tf.zeros(batch_size, dtype=tf.int32), depth=self.config.num_experts)
        if self.config.use_hard_routing:
            if training:
                return self._straight_through_categorical(
                    logits, self.config.gating_temperature
                )
            return tf.one_hot(tf.argmax(logits, axis=1), depth=self.config.num_experts)
        return tf.nn.softmax(logits / max(self.config.gating_temperature, 1e-6), axis=1)

    def _proxy_weights(self, context: tf.Tensor, *, training: bool) -> tf.Tensor:
        if self.config.use_proxy_weights:
            return tf.nn.softmax(self.proxy_network(context, training=training), axis=1)
        batch_size = tf.shape(context)[0]
        return tf.fill(
            [batch_size, self.config.num_objectives],
            1.0 / float(self.config.num_objectives),
        )

    def _all_expert_outputs(
        self,
        features: tf.Tensor,
        context: tf.Tensor,
        seq_obs: tf.Tensor,
        *,
        use_target: bool,
        training: bool,
    ) -> Tuple[tf.Tensor, tf.Tensor]:
        actors = self.target_expert_actors if use_target else self.expert_actors
        route_logits: List[tf.Tensor] = []
        continuous_outputs: List[tf.Tensor] = []
        for expert_index, actor in enumerate(actors):
            if expert_index == 2:
                route, cont = actor([seq_obs, features, context], training=training)
            else:
                route, cont = actor([features, context], training=training)
            route_logits.append(route)
            continuous_outputs.append(cont)
        return tf.stack(route_logits, axis=1), tf.stack(continuous_outputs, axis=1)

    def _project_continuous(self, continuous_raw: tf.Tensor) -> tf.Tensor:
        return 0.5 * (tf.tanh(continuous_raw) + 1.0)

    def _straight_through_categorical(
        self, logits: tf.Tensor, temperature: float
    ) -> tf.Tensor:
        gumbel_noise = -tf.math.log(
            -tf.math.log(tf.random.uniform(tf.shape(logits), 1e-8, 1.0 - 1e-8))
        )
        soft = tf.nn.softmax((logits + gumbel_noise) / max(temperature, 1e-6), axis=1)
        hard = tf.one_hot(tf.argmax(soft, axis=1), depth=soft.shape[-1])
        return tf.stop_gradient(hard - soft) + soft

    def _build_shared_encoder(self) -> keras.Model:
        obs_input = keras.Input(shape=(self.config.observation_dim,))
        x = layers.Dense(128, activation="relu")(obs_input)
        x = layers.Dense(self.config.shared_dim, activation="relu")(x)
        return keras.Model(obs_input, x, name="shared_encoder")

    def _build_gating_network(self) -> keras.Model:
        joint_input = keras.Input(shape=(self.config.shared_dim + self.config.context_dim,))
        x = layers.Dense(64, activation="relu")(joint_input)
        logits = layers.Dense(self.config.num_experts)(x)
        return keras.Model(joint_input, logits, name="gating_network")

    def _build_proxy_network(self) -> keras.Model:
        context_input = keras.Input(shape=(self.config.context_dim,))
        x = layers.Dense(32, activation="relu")(context_input)
        logits = layers.Dense(self.config.num_objectives)(x)
        return keras.Model(context_input, logits, name="proxy_network")

    def _build_expert_actors(self) -> List[keras.Model]:
        actors = [
            self._build_mlp_expert("latency_actor"),
            self._build_mlp_expert("energy_actor"),
            self._build_reliability_expert("reliability_actor"),
            self._build_mlp_expert("privacy_actor"),
        ]
        return actors

    def _build_mlp_expert(self, name: str) -> keras.Model:
        feature_input = keras.Input(shape=(self.config.shared_dim,))
        context_input = keras.Input(shape=(self.config.context_dim,))
        x = layers.Concatenate()([feature_input, context_input])
        x = layers.Dense(self.config.actor_hidden_dim, activation="relu")(x)
        x = layers.Dense(self.config.actor_hidden_dim, activation="relu")(x)
        route_logits = layers.Dense(self.config.num_routes)(x)
        continuous = layers.Dense(3)(x)
        return keras.Model([feature_input, context_input], [route_logits, continuous], name=name)

    def _build_reliability_expert(self, name: str) -> keras.Model:
        seq_input = keras.Input(
            shape=(self.config.sequence_length, self.config.observation_dim)
        )
        feature_input = keras.Input(shape=(self.config.shared_dim,))
        context_input = keras.Input(shape=(self.config.context_dim,))
        seq_features = layers.LSTM(32)(seq_input)
        x = layers.Concatenate()([seq_features, feature_input, context_input])
        x = layers.Dense(self.config.actor_hidden_dim, activation="relu")(x)
        route_logits = layers.Dense(self.config.num_routes)(x)
        continuous = layers.Dense(3)(x)
        return keras.Model(
            [seq_input, feature_input, context_input],
            [route_logits, continuous],
            name=name,
        )

    def _build_critic(self, name: str) -> keras.Model:
        feature_input = keras.Input(shape=(self.config.shared_dim,))
        route_input = keras.Input(shape=(self.config.num_routes,))
        continuous_input = keras.Input(shape=(3,))
        context_input = keras.Input(shape=(self.config.context_dim,))
        x = layers.Concatenate()(
            [feature_input, route_input, continuous_input, context_input]
        )
        x = layers.Dense(self.config.critic_hidden_dim, activation="relu")(x)
        x = layers.Dense(self.config.critic_hidden_dim, activation="relu")(x)
        q_value = layers.Dense(1)(x)
        return keras.Model(
            [feature_input, route_input, continuous_input, context_input],
            q_value,
            name=name,
        )
