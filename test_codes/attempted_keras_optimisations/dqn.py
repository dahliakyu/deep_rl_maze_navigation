import numpy as np
import random
from collections import deque
from keras.models import Sequential
from keras.layers import Dense, Input, Dropout, BatchNormalization
from keras.optimizers import Adam  
from keras.regularizers import l1_l2, l2
import tensorflow as tf
from util import RunningStats
import os
class DQNAgent:

    def __init__(self, state_size, action_size, learning_rate=0.001, gamma=0.95,
                 epsilon=1.0, epsilon_decay=0.995, epsilon_min=0.05,
                 replay_buffer_size=20000,  # Increased replay buffer
                 batch_size=128,          # Increased batch size
                 target_update_freq=100): # Update target network every 10 steps
        self.state_size = state_size
        self.action_size = action_size
        self.learning_rate = learning_rate
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.epsilon_min = epsilon_min
#        self.reward_stats = RunningStats()
        self.eps = 1e-8  # Numerical stability
        # Memory buffer
        self.replay_buffer = deque(maxlen=replay_buffer_size) # Use provided size
        self.batch_size = batch_size  # Use provided batch size
        self.target_update_freq = target_update_freq
        self.train_step = 0 # Counter for training steps

        # Create main and target networks
        self.q_network = self._build_model()
        self.target_network = self._build_model()
        self.update_target_network()  # Initialize target network weights
    
    def _process_reward(self, reward):
        self.reward_stats.update(reward)
        return (reward - self.reward_stats.mean) / (self.reward_stats.std + self.eps)
    
    def _build_model(self):
        model = Sequential()
        model.add(Input(shape=(self.state_size,)))
        
        # First hidden layer with combined regularization
        model.add(Dense(32, 
                    activation='relu',
#                    kernel_regularizer=l1_l2(l1=0.0005, l2=0.001),  # L1+L2 reg
#                    bias_regularizer=l2(0.001)
        ))
#        model.add(BatchNormalization())  # Add batch norm
#        model.add(Dropout(0.3))  # Add dropout
        
        # Second hidden layer
        model.add(Dense(32,
                    activation='relu',
#                    kernel_regularizer=l1_l2(l1=0.0005, l2=0.001),
#                    bias_regularizer=l2(0.001)
    	))
#        model.add(BatchNormalization())
#        model.add(Dropout(0.2))  # Less dropout in deeper layers
        
        # Output layer
        model.add(Dense(self.action_size, activation='linear'))
        
        # Optimizer with adjusted parameters
        optimizer = Adam(learning_rate=self.learning_rate,
                        clipnorm=1.0)  # Add value clipping
        
        model.compile(optimizer=optimizer,
                    loss='huber',  # Explicit huber loss
                    metrics=['mae'])  # Track mean absolute error
        return model

    def store_experience(self, state, action, reward, next_state, done):
#        processed_reward = self._process_reward(reward)
        self.replay_buffer.append((state, action, reward, next_state, done))

    def act(self, state):
        if np.random.rand() <= self.epsilon:
            return random.randrange(self.action_size)

        state = np.array(state).reshape(1, -1)  # Reshape for Keras input
        q_values = self.q_network.predict(state, verbose=0)
        return np.argmax(q_values[0])


    def replay(self):
        if len(self.replay_buffer) < self.batch_size:
            return

        # Sample random batch
        batch = random.sample(self.replay_buffer, self.batch_size)
        states = np.array([i[0] for i in batch])
        actions = np.array([i[1] for i in batch])
        rewards = np.array([i[2] for i in batch])
        next_states = np.array([i[3] for i in batch])
        dones = np.array([i[4] for i in batch])

        # Calculate targets - Vectorized for efficiency
        next_q_values = self.target_network.predict(next_states, verbose=0)
        targets = rewards + self.gamma * np.max(next_q_values, axis=1) * (1 - dones) # Vectorized

        # Predict Q values for the current state and action
        q_values = self.q_network.predict(states, verbose=0)

        # Update the Q values for the actions taken
        for i in range(self.batch_size):
            q_values[i][actions[i]] = targets[i]

        # Train the network
        self.q_network.fit(states, q_values, epochs=1, verbose=0)
        
        self.train_step += 1
        if self.train_step % self.target_update_freq == 0:
            self.update_target_network()  # Update target network

    def update_target_network(self):
        self.target_network.set_weights(self.q_network.get_weights())

    def update_epsilon(self):
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)