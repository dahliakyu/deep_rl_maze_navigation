import itertools
from rl_algorithms.dqn import DQNAgent
from experiments.dqn_experiment import train_dqn_agent
from maze_env.environment import ComplexMazeEnv
import json

# Define hyperparameter ranges
learning_rates = [0.002]
gammas = [0.90, 0.95, 0.99]
epsilon_decays = [0.95]

# Create all combinations of hyperparameters
param_combinations = list(itertools.product(learning_rates, gammas, epsilon_decays))

for lr, gamma, epsilon_decay in param_combinations:
    print(f"Training with lr={lr}, gamma={gamma}, epsilon_decay={epsilon_decay}")

    # Initialize environment and agent with current hyperparameters
    env = ComplexMazeEnv(maze_file='./generated_mazes/maze_9_9_hard.json')
    state_size = 2
    action_size = 4

    agent = DQNAgent(
        state_size=state_size,
        action_size=action_size,
        learning_rate=lr,
        gamma=gamma,
        epsilon=1.0,
        epsilon_decay=epsilon_decay,
        epsilon_min=0.05,
    )

    # Train the agent
    rewards, steps = train_dqn_agent(env, agent, num_episodes=1000)  # Reduce episodes for testing

    # Save all data to a single JSON file after training is complete for this hyperparameter set.
    filename = f"all_episodes_lr_{lr}_gamma_{gamma}_decay_{epsilon_decay}.json"
    data_to_save = {
        'learning_rate': lr,
        'gamma': gamma,
        'epsilon_decay': epsilon_decay,
        'reward_history': rewards,
        'steps_history': steps
    }

    with open(filename, 'w') as f:
        json.dump(data_to_save, f, indent=4)

    print(f"All episodes saved to {filename}")