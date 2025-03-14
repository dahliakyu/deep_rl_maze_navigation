import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import random
from collections import deque
import torch.nn.functional as F

class DQN(nn.Module):
    def __init__(self, input_size, output_size):
        super(DQN, self).__init__()
        self.fc1 = nn.Linear(input_size, 128)
        self.fc2 = nn.Linear(128, 128)
        self.fc3 = nn.Linear(128, output_size)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return self.fc3(x)

class DQNAgent:
    def __init__(self, state_size, action_size, learning_rate=0.001, gamma=0.99, epsilon=1.0, epsilon_decay=0.995, epsilon_min=0.1):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.state_size = state_size #state tuples (x,y) so 2
        self.action_size = action_size # actions i can take so 4
        self.learning_rate = learning_rate # learning rate of the model
        self.gamma = gamma # Discount factor for future rewards.
        self.epsilon = epsilon #Exploration rate (probability of taking a random action).
        self.epsilon_decay = epsilon_decay #switch from exploration to remembering
        self.epsilon_min = epsilon_min #lowest the remembering should go
        self.q_network = DQN(state_size, action_size).to(self.device) # two networks for stable training (chatgpt help)
        self.target_network = DQN(state_size, action_size).to(self.device)
        self.optimizer = optim.Adam(self.q_network.parameters(), lr=self.learning_rate) #optimizer
        self.replay_buffer = deque(maxlen=50_000) #memory limiter
        self.batch_size = 64 #when to start training on batches

    def act(self, state):
        state = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        if np.random.rand() <= self.epsilon:
            return random.randrange(self.action_size)  # Random action (exploration)
        with torch.no_grad():
            q_values = self.q_network(state)
        return torch.argmax(q_values).item()  # Action with the highest Q-value (exploitation)

    def store_experience(self, state, action, reward, next_state, done):
        self.replay_buffer.append((state, action, reward, next_state, done))

    def replay(self):
        if len(self.replay_buffer) < self.batch_size:
            return
        
        batch = random.sample(self.replay_buffer, self.batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)

        # Convert lists of numpy arrays to single numpy arrays
        states = torch.FloatTensor(np.stack(states)).to(self.device)
        actions = torch.LongTensor(np.stack(actions)).to(self.device)
        rewards = torch.FloatTensor(np.stack(rewards)).to(self.device)
        next_states = torch.FloatTensor(np.stack(next_states)).to(self.device)
        dones = torch.BoolTensor(np.stack(dones)).to(self.device)

        # Rest of the code remains unchanged
        current_q_values = self.q_network(states).gather(1, actions.unsqueeze(1))
        next_q_values = self.target_network(next_states).max(1)[0].detach()
        targets = rewards + (self.gamma * next_q_values * ~dones)
        loss = F.mse_loss(current_q_values.squeeze(), targets)
        
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

    def update_target_network(self):
        self.target_network.load_state_dict(self.q_network.state_dict())

    def update_epsilon(self):
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay
