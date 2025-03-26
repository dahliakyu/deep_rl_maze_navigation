import numpy as np
import random
from collections import deque
import tensorflow as tf
from tensorflow import keras
from keras.models import Sequential
from keras.layers import Dense, Input
from keras.optimizers import Adam
from util import RunningStats

class DQNAgent:

    def __init__(self, state_size, action_size, learning_rate=0.001, gamma=0.95,
                 epsilon=1.0, epsilon_decay=0.995, epsilon_min=0.05,
                 replay_buffer_size=20000, batch_size=128, target_update_freq=100):
        self.state_size = state_size
        self.action_size = action_size
        self.learning_rate = learning_rate
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.epsilon_min = epsilon_min
        self.eps = 1e-8
        self.replay_buffer = deque(maxlen=replay_buffer_size)
        self.batch_size = batch_size
        self.target_update_freq = target_update_freq
        self.train_step = 0

        # Initialize networks
        self.q_network = self._build_model()
        self.target_network = self._build_model()
        self.update_target_network()

        # Initialize reward normalizer
        self.reward_stats = RunningStats()

    def _build_model(self):
        model = Sequential([
            Input(shape=(self.state_size,)),
            Dense(32, activation='relu'),
            Dense(32, activation='relu'),
            Dense(self.action_size, activation='linear')
        ])
        model.compile(optimizer=Adam(learning_rate=self.learning_rate, clipnorm=1.0),
                      loss='huber',
                      metrics=['mae'])
        return model

    def _process_reward(self, reward):
        self.reward_stats.update(reward)
        return (reward - self.reward_stats.mean) / (self.reward_stats.std + self.eps)

    def store_experience(self, state, action, reward, next_state, done):
        self.replay_buffer.append((state, action, reward, next_state, done))

    def act(self, state):
        if np.random.rand() <= self.epsilon:
            return random.randrange(self.action_size)
        state_tensor = tf.convert_to_tensor([state], dtype=tf.float32)
        q_values = self._predict(state_tensor)
        return tf.argmax(q_values[0]).numpy()

    @tf.function
    def _predict(self, state):
        return self.q_network(state)

    def replay(self):
        if len(self.replay_buffer) < self.batch_size:
            return

        batch = random.sample(self.replay_buffer, self.batch_size)
        states = np.array([x[0] for x in batch], dtype=np.float32)
        actions = np.array([x[1] for x in batch], dtype=np.int32)
        rewards = np.array([x[2] for x in batch], dtype=np.float32)
        next_states = np.array([x[3] for x in batch], dtype=np.float32)
        dones = np.array([x[4] for x in batch], dtype=np.float32)

        # Convert to tensors and execute training step
        states_tensor = tf.convert_to_tensor(states, dtype=tf.float32)
        actions_tensor = tf.convert_to_tensor(actions, dtype=tf.int32)
        rewards_tensor = tf.convert_to_tensor(rewards, dtype=tf.float32)
        next_states_tensor = tf.convert_to_tensor(next_states, dtype=tf.float32)
        dones_tensor = tf.convert_to_tensor(dones, dtype=tf.float32)

        self._train_step(states_tensor, actions_tensor, rewards_tensor, 
                         next_states_tensor, dones_tensor)

        self.train_step += 1
        if self.train_step % self.target_update_freq == 0:
            self.update_target_network()

    @tf.function
    def _train_step(self, states, actions, rewards, next_states, dones):
        with tf.GradientTape() as tape:
            # Calculate target Q-values
            next_q = self.target_network(next_states)
            max_next_q = tf.reduce_max(next_q, axis=1)
            targets = rewards + self.gamma * max_next_q * (1 - dones)

            # Calculate current Q-values and gather those for the taken actions
            current_q = self.q_network(states)
            action_mask = tf.one_hot(actions, self.action_size, dtype=tf.float32)
            selected_q = tf.reduce_sum(current_q * action_mask, axis=1)

            # Compute loss
            loss = tf.keras.losses.Huber()(targets, selected_q)

        # Apply gradients
        gradients = tape.gradient(loss, self.q_network.trainable_variables)
        self.q_network.optimizer.apply_gradients(zip(gradients, self.q_network.trainable_variables))

    def update_target_network(self):
        self.target_network.set_weights(self.q_network.get_weights())

    def update_epsilon(self):
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)