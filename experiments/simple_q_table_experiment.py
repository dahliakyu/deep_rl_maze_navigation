import matplotlib.pyplot as plt
from maze_env.environment import MazeEnv
from rl_algorithms.simple_q_table import QLearningAgent
from visualization import Visualization
import copy

def run_q_learning_experiment(env, num_episodes=1000, learning_rate=0.1, discount_factor=0.9, exploration_rate=0.1, snapshot_interval=200):
    """
    Runs a Q-learning experiment on the given maze environment.

    Args:
        env (MazeEnv): The maze environment.
        num_episodes (int): The number of episodes to train the agent.
        learning_rate (float): The learning rate (alpha).
        discount_factor (float): The discount factor (gamma).
        exploration_rate (float): The exploration rate (epsilon).
        snapshot_interval (int): Interval for saving Q-table snapshots.
    """

    agent = QLearningAgent(env, learning_rate=learning_rate, discount_factor=discount_factor, exploration_rate=exploration_rate)
    vis = Visualization(env, agent)

    rewards_per_episode = []
    q_table_snapshots = []  # List to store Q-table snapshots

    for episode in range(num_episodes):
        state = env.reset()
        done = False
        total_reward = 0

        while not done:
            action = agent.choose_action(state)
            next_state, reward, done = env.step(action)
            agent.learn(state, action, reward, next_state, done)
            total_reward += reward
            state = next_state

        rewards_per_episode.append(total_reward)

        if (episode + 1) % 100 == 0:
            print(f"Episode {episode + 1}/{num_episodes}, Total Reward: {total_reward}")

        # Take snapshots of the Q-table at certain intervals
        if (episode + 1) % snapshot_interval == 0:
            q_table_snapshots.append((episode + 1, copy.deepcopy(agent.q_table))) # store a copy

    # Visualize learning progress
    vis.visualize_learning(rewards_per_episode)

    # Visualize final results
    vis.visualize_final_result()

    # Visualize Q-values in 3D
    vis.visualize_q_values_3d()

    # Visualize learning snapshots
    vis.visualize_learning_snapshots(q_table_snapshots)

    # Print the learned Q-table to the console
    print("Learned Q-table:")
    print(agent.q_table)

if __name__ == '__main__':
    # Define a custom maze
    custom_walls = [(1, 1), (1, 2), (1, 3), (3, 1), (3, 3)]
    custom_start = (0, 0)
    custom_goal = (4, 4)
    env = MazeEnv(grid_size=(5, 5), walls=custom_walls, start=custom_start, goal=custom_goal)

    run_q_learning_experiment(env)