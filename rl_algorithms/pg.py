import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical
import numpy as np

class PolicyNetwork(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(PolicyNetwork, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, output_size)
    
    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return F.softmax(x, dim=1)

class PolicyGradientAgent:
    def __init__(self, maze_size, hidden_size=128, lr=0.01, gamma=0.99):
        self.maze_size = maze_size
        self.policy_network = PolicyNetwork(input_size=2, hidden_size=hidden_size, output_size=4)
        self.optimizer = torch.optim.Adam(self.policy_network.parameters(), lr=lr)
        self.gamma = gamma
    
    def get_action(self, state):
        x, y = state
        normalized_x = x / (self.maze_size - 1)
        normalized_y = y / (self.maze_size - 1)
        state_tensor = torch.FloatTensor([normalized_x, normalized_y]).unsqueeze(0)
        probs = self.policy_network(state_tensor)
        action_dist = Categorical(probs)
        action = action_dist.sample()
        log_prob = action_dist.log_prob(action)
        return action.item(), log_prob
    
    def update_policy(self, log_probs, rewards):
        returns = self._compute_returns(rewards)
        loss = -(torch.stack(log_probs) * returns).sum()
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
    
    def _compute_returns(self, rewards):
        G = 0
        returns = []
        for r in reversed(rewards):
            G = r + self.gamma * G
            returns.insert(0, G)
        returns = torch.tensor(returns, dtype=torch.float32)
        returns = (returns - returns.mean()) / (returns.std() + 1e-9)
        return returns

