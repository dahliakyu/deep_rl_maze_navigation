import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import random
from collections import deque
import torch.nn.functional as F
import matplotlib.pyplot as plt
import matplotlib.patches as patches

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
        self.replay_buffer = deque(maxlen=10000) #memory limiter
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

        states = torch.FloatTensor(states).to(self.device)
        actions = torch.LongTensor(actions).to(self.device)
        rewards = torch.FloatTensor(rewards).to(self.device)
        next_states = torch.FloatTensor(next_states).to(self.device)
        dones = torch.BoolTensor(dones).to(self.device)

        # Get Q-values from current network
        current_q_values = self.q_network(states).gather(1, actions.unsqueeze(1))

        # Get max Q-values from target network for next state
        next_q_values = self.target_network(next_states).max(1)[0].detach()

        # Calculate target Q-values
        targets = rewards + (self.gamma * next_q_values * ~dones)

        # Compute loss
        loss = F.mse_loss(current_q_values.squeeze(), targets)

        # Optimize the model
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

    def update_target_network(self):
        self.target_network.load_state_dict(self.q_network.state_dict())

    def update_epsilon(self):
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

class MazeEnv:
    def __init__(self):
        self.maze = np.zeros((5, 5))
        self.maze[1, 1:4] = 1  # Walls
        self.maze[3, [1, 3]] = 1
        self.maze[4, 4] = 9  # Goal
        self.start = (0, 0)
        self.current_state = self.start
        self.action_space = [0, 1, 2, 3]  # Up, Down, Left, Right
    
    def reset(self):
        self.current_state = self.start
        return self.current_state
    
    def step(self, action):
        x, y = self.current_state
        if action == 0 and x > 0 and self.maze[x - 1, y] != 1:
            x -= 1  # Up
        elif action == 1 and x < 4 and self.maze[x + 1, y] != 1:
            x += 1  # Down
        elif action == 2 and y > 0 and self.maze[x, y - 1] != 1:
            y -= 1  # Left
        elif action == 3 and y < 4 and self.maze[x, y + 1] != 1:
            y += 1  # Right
        
        self.current_state = (x, y)
        if self.maze[x, y] == 9:
            return self.current_state, 10, True  # Goal reached
        return self.current_state, -0.1, False  # Small penalty for moving
    
    def render(self, path=None):
        fig, ax = plt.subplots(figsize=(5, 5))
        ax.imshow(self.maze, cmap='gray_r')
        for i in range(6):
            ax.axhline(i - 0.5, color='black', linewidth=1)
            ax.axvline(i - 0.5, color='black', linewidth=1)
        ax.text(0, 0, 'S', ha='center', va='center', fontsize=14, color='blue')
        ax.text(4, 4, 'G', ha='center', va='center', fontsize=14, color='red')
        if path:
            for (x, y) in path:
                ax.add_patch(patches.Circle((y, x), 0.2, color='green'))
        plt.show()

def train(env, agent, episodes=1000, max_steps=100, target_update_freq=10):
    episode_rewards = []
    episode_lengths = []
    best_path = None
    best_reward = -float('inf')

    for episode in range(episodes):
        state = env.reset()
        total_reward = 0
        path = [state]

        for step in range(max_steps):
            action = agent.act(state)
            next_state, reward, done = env.step(action)
            agent.store_experience(state, action, reward, next_state, done)
            state = next_state
            total_reward += reward
            path.append(state)

            if done:
                break

        # Train the agent using experience replay
        agent.replay()

        # Update target network periodically
        if episode % target_update_freq == 0:
            agent.update_target_network()

        # Decay epsilon
        agent.update_epsilon()

        # Log episode results
        episode_rewards.append(total_reward)
        episode_lengths.append(step + 1)

        # Track the best path
        if total_reward > best_reward:
            best_reward = total_reward
            best_path = path

        # Print progress
        if (episode + 1) % 100 == 0:
            print(f"Episode {episode + 1}/{episodes}, Reward: {total_reward}, Epsilon: {agent.epsilon:.2f}")



    # Render the best path
    print("Best Path Reward:", best_reward)
    env.render(best_path)

# Initialize environment and agent
env = MazeEnv()
state_size = 2  # (x, y) coordinates
action_size = 4  # Up, Down, Left, Right
agent = DQNAgent(state_size, action_size)

# Train the agent
train(env, agent, episodes=1000)