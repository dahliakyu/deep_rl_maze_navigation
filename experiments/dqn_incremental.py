import torch
from maze_env.maze_generation import generate_maze, maze_to_numpy
from maze_env.environment import ComplexMazeEnv
from experiments.dqn_experiment import train_dqn_agent
from rl_algorithms.dqn import DQNAgent
import numpy as np
import json

# 1. Modified ComplexMazeEnv Class (state normalization)
class ComplexMazeEnv:
    def __init__(self, maze_array):
        self.maze = maze_array
        self.size = self.maze.shape[0]
        self.start = (0, 0)
        self.goal = (self.size-1, self.size-1)
        self.current_state = self.start

    def reset(self):
        self.current_state = self.start
        return self._normalize_state(self.current_state)

    def step(self, action):
        x, y = self.current_state
        new_x, new_y = x, y

        if action == 0:  # Up
            new_x = max(0, x-1)
        elif action == 1:  # Down
            new_x = min(self.size-1, x+1)
        elif action == 2:  # Left
            new_y = max(0, y-1)
        elif action == 3:  # Right
            new_y = min(self.size-1, y+1)

        # Wall collision check
        if self.maze[new_x, new_y] != 1:
            self.current_state = (new_x, new_y)

        # Calculate reward
        reward = -0.01  # Step penalty

        # Proximity reward (uncommented)
        distance_before = np.linalg.norm(np.array((x, y)) - np.array(self.goal))
        distance_after = np.linalg.norm(np.array(self.current_state) - np.array(self.goal))
        if distance_after < distance_before:
            reward += 0.1 * (distance_before - distance_after)

        if self.current_state == self.goal:
            reward = 100
            done = True
        elif self.maze[new_x, new_y] == 1:
            reward = -5
            done = False
        else:
            done = False

        return self._normalize_state(self.current_state), reward, done

    def _normalize_state(self, state):
        x, y = state
        return (x / (self.size - 1), y / (self.size - 1))
    
# 2. Training Function
def train_incrementally():
    all_rewards = []
    all_steps = []
    # Initialize agent
    agent = DQNAgent(
        state_size=2,
        action_size=4,
        learning_rate=0.002,
        gamma=0.98,
        epsilon_decay=0.9995,
        replay_buffer_size=20000
    )
    training_history = []
    # Training sequence
    sizes = [5]
    for size in sizes:
        print(f"\n=== Training on {size}x{size} mazes ===")
        
        # Gradually reduce maze complexity
        for complexity in range(10):
            extra_passages = 9 - complexity
            episode_rewards = []
            episode_steps = []

            # Generate 10 mazes per complexity level
            for maze_num in range(10):
                # Generate and convert maze
                maze_structure = generate_maze(size, size, extra_passages)
                numpy_maze = maze_to_numpy(maze_structure)
                env = ComplexMazeEnv(numpy_maze)

                # Train on this maze for 10 episodes
                rewards, steps = train_dqn_agent(
                    env=env,
                    agent=agent,
                    num_episodes=50,  # 50 episodes per maze
                    max_steps=size*200  # Scale steps with maze size
                )

                filename = f"incr_comp_{complexity}_{maze_num}.json"
                data_to_save = {
                    'maze_config': numpy_maze.tolist(),
                    'reward_history': rewards,
                    'steps_history': steps
                }

                with open(filename, 'w') as f:
                    json.dump(data_to_save, f, indent=4)

                print(f"All episodes saved to {filename}")
                episode_rewards.extend(rewards)
                episode_steps.extend(steps)
            # Save progress for this complexity level
            all_rewards.extend(episode_rewards)
            all_steps.extend(episode_steps)
            print(f"Size {size}x{size} | Complexity {complexity+1}/10 | Avg Reward: {np.mean(episode_rewards):.2f}")
                    
        # Save checkpoint and training history
        torch.save(agent.q_network.state_dict(), f"dqn_{size}x{size}.pth")

# 3. Execute Training
if __name__ == "__main__":
    train_incrementally()