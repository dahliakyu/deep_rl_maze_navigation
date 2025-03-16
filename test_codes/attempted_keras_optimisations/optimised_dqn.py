import numpy as np
import random
from collections import deque
import tensorflow as tf
from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import Adam


#        Enable memory growth for GPU if available
gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    for gpu in gpus:
        tf.config.experimental.set_memory_growth(gpu, True)

# Set up threading for CPU
tf.config.threading.set_inter_op_parallelism_threads(4)
tf.config.threading.set_intra_op_parallelism_threads(4)
class DQNAgent:
    def __init__(self, state_size, action_size, learning_rate=0.005, gamma=0.95,
                 epsilon=1.0, epsilon_decay=0.98, epsilon_min=0.01,
                 replay_buffer_size=5000,
                 batch_size=32,
                 target_update_freq=20):
        self.state_size = state_size
        self.action_size = action_size
        self.learning_rate = learning_rate
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.epsilon_min = epsilon_min
        
        # Memory buffer
        self.replay_buffer = deque(maxlen=replay_buffer_size)
        self.batch_size = batch_size
        self.target_update_freq = target_update_freq
        self.train_step = 0
        
        # Create optimized networks for small mazes
        self.q_network = self._build_model()
        self.target_network = self._build_model()
        self.update_target_network()
        

    
    def _build_model(self):
        # Ultra-simple model for a 5x5 grid - just enough complexity
        model = Sequential([
            Dense(16, activation='relu', input_shape=(self.state_size,)),
            Dense(16, activation='relu'),
            Dense(self.action_size, activation='linear')
        ])
        
        model.compile(
            loss='mse',
            optimizer=Adam(learning_rate=self.learning_rate)
        )
        return model

    def store_experience(self, state, action, reward, next_state, done):
        self.replay_buffer.append((state, action, reward, next_state, done))

    def act(self, state, training=True):
        if training and np.random.rand() <= self.epsilon:
            return random.randrange(self.action_size)

        state_tensor = np.array(state, dtype=np.float32).reshape(1, -1)
        q_values = self.q_network.predict(state_tensor, verbose=0)
        return np.argmax(q_values[0])

    def replay(self):
        if len(self.replay_buffer) < self.batch_size:
            return 0  # Return loss for tracking

        # Sample minibatch
        minibatch = random.sample(self.replay_buffer, self.batch_size)
        
        # Prepare batch data
        states = np.vstack([exp[0] for exp in minibatch])
        actions = np.array([exp[1] for exp in minibatch])
        rewards = np.array([exp[2] for exp in minibatch])
        next_states = np.vstack([exp[3] for exp in minibatch])
        dones = np.array([exp[4] for exp in minibatch])
        
        # Calculate target Q values efficiently
        next_q_values = self.target_network.predict(next_states, verbose=0)
        target_q_values = rewards + (1 - dones) * self.gamma * np.max(next_q_values, axis=1)
        
        # Get current predictions
        current_q = self.q_network.predict(states, verbose=0)
        
        # Update only the actions taken
        for i, action in enumerate(actions):
            current_q[i][action] = target_q_values[i]
            
        # Perform batch training with a single call
        history = self.q_network.fit(states, current_q, epochs=1, verbose=0, batch_size=self.batch_size)
        
        # Check if it's time to update target network
        self.train_step += 1
        if self.train_step % self.target_update_freq == 0:
            self.update_target_network()
            
        return history.history['loss'][0] if 'loss' in history.history else 0

    def update_target_network(self):
        self.target_network.set_weights(self.q_network.get_weights())

    def update_epsilon(self):
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)