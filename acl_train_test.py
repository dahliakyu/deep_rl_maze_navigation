import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from collections import deque
import random
import matplotlib.pyplot as plt
from typing import List, Tuple, Optional
import time

class LineEnvironment:
    def __init__(self, length=5):
        self.length = length
        self.reset()

    def reset(self):
        self.position = 0
        self.steps = 0
        self.max_steps = self.length * 2
        return self._get_state()

    def _get_state(self):
        # Normalized position and goal
        if self.length == 1:
            current_pos = 0.0
            goal_pos = 0.0
        else:
            current_pos = self.position / (self.length - 1)
            goal_pos = 1.0
        return np.array([current_pos, goal_pos], dtype=np.float32)

    def step(self, action):
        self.steps += 1
        # Action: 0 = left, 1 = right
        if action == 1:  # Move right
            self.position = min(self.position + 1, self.length - 1)
        else:  # Move left
            self.position = max(self.position - 1, 0)

        done = (self.position == self.length - 1) or (self.steps >= self.max_steps)
        reward = 1.0 if self.position == self.length - 1 else -0.01
        return self._get_state(), reward, done

class SimpleNetwork(nn.Module):
    def __init__(self, input_size, output_size):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(input_size, 32),
            nn.ReLU(),
            nn.Linear(32, output_size))

    def forward(self, x):
        return self.network(x)

class SimpleCurriculumTrainer:
    def __init__(self, initial_length=3, max_length=100):
        self.initial_length = initial_length
        self.max_length = max_length
        self.current_length = initial_length
        self.success_threshold = 0.8
        self.success_window = deque(maxlen=50)

    def train(self, episodes=2000) -> nn.Module:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        net = SimpleNetwork(2, 2).to(device)
        optimizer = optim.Adam(net.parameters(), lr=0.001)

        buffer = deque(maxlen=1000)
        epsilon = 1.0

        for episode in range(episodes):
            env = LineEnvironment(self.current_length)
            state = env.reset()
            total_reward = 0
            done = False

            while not done:
                state_t = torch.FloatTensor(state).unsqueeze(0).to(device)

                if random.random() < epsilon:
                    action = random.randint(0, 1)
                else:
                    with torch.no_grad():
                        q_vals = net(state_t)
                        action = torch.argmax(q_vals).item()

                next_state, reward, done = env.step(action)
                buffer.append((state, action, reward, next_state, done))
                state = next_state
                total_reward += reward

                if len(buffer) >= 32:
                    batch = random.sample(buffer, 32)
                    self._train_step(net, batch, optimizer, device)

            self.success_window.append(1 if total_reward > 0 else 0)
            success_rate = sum(self.success_window) / len(self.success_window) if self.success_window else 0.0

            # Curriculum progression
            if (len(self.success_window) == self.success_window.maxlen and
                success_rate >= self.success_threshold and
                self.current_length < self.max_length):
                self.current_length += 1
                self.success_window.clear()
                epsilon = 1.0  # Reset exploration rate
                print(f"Progressing to length {self.current_length}")

            epsilon = max(0.01, epsilon * 0.995)

            if episode % 100 == 0:
                print(f"Episode {episode}, Success Rate: {success_rate:.2f}, Length: {self.current_length}")

        return net

    def _train_step(self, net, batch, optimizer, device):
        states, actions, rewards, next_states, dones = zip(*batch)

        states_t = torch.FloatTensor(np.array(states)).to(device)
        next_states_t = torch.FloatTensor(np.array(next_states)).to(device)
        actions_t = torch.LongTensor(actions).to(device)
        rewards_t = torch.FloatTensor(rewards).to(device)
        done_mask = torch.BoolTensor(dones).to(device)

        current_q = net(states_t).gather(1, actions_t.unsqueeze(-1)).squeeze(-1)

        with torch.no_grad():
            next_q = net(next_states_t).max(1)[0]
            next_q[done_mask] = 0.0
            expected_q = rewards_t + 0.99 * next_q

        loss = nn.MSELoss()(current_q, expected_q)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

class BaseTrainer:
    def __init__(self, length: int = 10):
        self.length = length

    def _train_step(self, net: nn.Module, batch: List[Tuple], optimizer: optim.Optimizer, device: torch.device):
        states, actions, rewards, next_states, dones = zip(*batch)

        states_t = torch.FloatTensor(np.array(states)).to(device)
        next_states_t = torch.FloatTensor(np.array(next_states)).to(device)
        actions_t = torch.LongTensor(actions).to(device)
        rewards_t = torch.FloatTensor(rewards).to(device)
        done_mask = torch.BoolTensor(dones).to(device)

        current_q = net(states_t).gather(1, actions_t.unsqueeze(-1)).squeeze(-1)

        with torch.no_grad():
            next_q = net(next_states_t).max(1)[0]
            next_q[done_mask] = 0.0
            expected_q = rewards_t + 0.99 * next_q

        loss = nn.MSELoss()(current_q, expected_q)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        return loss.item()

    def train(self, episodes: int = 2000) -> Tuple[List[float], List[float], nn.Module]:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        net = SimpleNetwork(2, 2).to(device)
        optimizer = optim.Adam(net.parameters(), lr=0.001)

        buffer = deque(maxlen=1000)
        epsilon = 1.0

        success_rates = []
        success_window = deque(maxlen=50)
        losses = []

        for episode in range(episodes):
            env = LineEnvironment(self.length)
            state = env.reset()
            total_reward = 0
            episode_losses = []

            while True:
                state_t = torch.FloatTensor(state).unsqueeze(0).to(device)

                if random.random() < epsilon:
                    action = random.randint(0, 1)
                else:
                    with torch.no_grad():
                        q_vals = net(state_t)
                        action = torch.argmax(q_vals).item()

                next_state, reward, done = env.step(action)
                buffer.append((state, action, reward, next_state, done))
                state = next_state
                total_reward += reward

                if len(buffer) >= 32:
                    loss = self._train_step(net, random.sample(buffer, 32), optimizer, device)
                    episode_losses.append(loss)

                if done:
                    break

            success_window.append(1 if total_reward > 0 else 0)
            success_rate = sum(success_window) / len(success_window) if success_window else 0
            success_rates.append(success_rate)
            losses.append(np.mean(episode_losses) if episode_losses else 0)

            epsilon = max(0.01, epsilon * 0.995)

            if episode % 100 == 0:
                print(f"Episode {episode}, Success Rate: {success_rate:.2f}")

        return success_rates, losses, net

def evaluate_model(trainer, num_eval_episodes: int = 100) -> float:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if isinstance(trainer, SimpleCurriculumTrainer):
        net = trainer.train()
        env_length = trainer.current_length # Use current_length for CurriculumTrainer
    else:
        _, _, net = trainer.train()
        env_length = trainer.length # Use length for BaseTrainer

    successes = 0
    for _ in range(num_eval_episodes):
        env = LineEnvironment(env_length)
        state = env.reset()
        done = False

        while not done:
            state_t = torch.FloatTensor(state).unsqueeze(0).to(device)
            with torch.no_grad():
                q_vals = net(state_t)
                action = torch.argmax(q_vals).item()

            next_state, reward, done = env.step(action)
            state = next_state

            if reward > 0:
                successes += 1
                break

    return successes / num_eval_episodes

def compare_methods(max_length: int = 10, num_trials: int = 3):
    curriculum_results = []
    standard_results = []

    for trial in range(num_trials):
        print(f"\nTrial {trial + 1}/{num_trials}")

        print("Training with curriculum learning...")
        curriculum_trainer = SimpleCurriculumTrainer(initial_length=3, max_length=max_length)
        start_time = time.time()
        curriculum_score = evaluate_model(curriculum_trainer)
        curriculum_time = time.time() - start_time
        curriculum_results.append((curriculum_score, curriculum_time))

        print("Training without curriculum learning...")
        standard_trainer = BaseTrainer(length=max_length)
        start_time = time.time()
        standard_score = evaluate_model(standard_trainer)
        standard_time = time.time() - start_time
        standard_results.append((standard_score, standard_time))

    curriculum_scores, curriculum_times = zip(*curriculum_results)
    standard_scores, standard_times = zip(*standard_results)

    print("\nResults Summary:")
    print(f"Curriculum Learning:")
    print(f"  Average Success Rate: {np.mean(curriculum_scores):.3f} ± {np.std(curriculum_scores):.3f}")
    print(f"  Average Training Time: {np.mean(curriculum_times):.1f}s ± {np.std(curriculum_times):.1f}s")

    print(f"\nStandard Learning:")
    print(f"  Average Success Rate: {np.mean(standard_scores):.3f} ± {np.std(standard_scores):.3f}")
    print(f"  Average Training Time: {np.mean(standard_times):.1f}s ± {np.std(standard_times):.1f}s")

if __name__ == "__main__":
    compare_methods(max_length=10, num_trials=3)
