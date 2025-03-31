import matplotlib.pyplot as plt
from maze_env.maze_generation import generate_maze, maze_to_numpy
from maze_env.environment import ComplexMazeEnv
from rl_algorithms.sarsa import SarsaAgent
import json
import itertools

# Hyperparameter ranges
alphas = [0.1, 0.2, 0.3]
gammas = [0.85, 0.90, 0.95]
epsilons = [0.1, 0.2, 0.3]
param_combinations = list(itertools.product(alphas, gammas, epsilons))

# Experiment configuration
COMPLEXITY_LEVELS = 8  # 0-15
MAZE_SIZE = (4, 4)      # Rows, columns
MAZES_PER_COMPLEXITY = 100
EPISODES = 500

def train_and_evaluate():
    # Iterate through all hyperparameter combinations
    for alpha, gamma, epsilon in param_combinations:
        print(f"\nTraining α={alpha}, γ={gamma}, ε={epsilon}")
        
        # Process each complexity level
        for complexity in range(COMPLEXITY_LEVELS):
            extra_passages = 7 - complexity  # Adjust complexity formula
            
            print(f" Complexity {complexity} (Extra Passages: {extra_passages})")
            
            rewards_history = []  # Stores results from all mazes
            steps_history = []

            # Generate and test on multiple mazes
            for _ in range(MAZES_PER_COMPLEXITY):
                # Generate maze
                maze_struct = generate_maze(MAZE_SIZE[0], MAZE_SIZE[1], extra_passages)
                numpy_maze = maze_to_numpy(maze_struct)
                env = ComplexMazeEnv(numpy_maze)
                agent = SarsaAgent(env, 4, alpha, gamma, epsilon)

                # Train agent
                maze_rewards, maze_steps = run_training_episodes(agent, env)
                rewards_history.append(maze_rewards)
                steps_history.append(maze_steps)

            # Calculate averages across mazes
            avg_rewards = [sum(ep)/MAZES_PER_COMPLEXITY for ep in zip(*rewards_history)]
            avg_steps = [sum(ep)/MAZES_PER_COMPLEXITY for ep in zip(*steps_history)]

            # Save results
            save_results(complexity, alpha, gamma, epsilon, 
                       avg_rewards, avg_steps, extra_passages)

def run_training_episodes(agent, env):
    episode_rewards = []
    episode_steps = []
    
    for ep in range(EPISODES):
        state = env.reset()
        total_reward = 0
        steps = 0
        done = False
        action = agent.choose_action(state)  # Initial action
        while not done:
            # Take action, observe next state and reward
            next_state, reward, done = env.step(action)

            # SARSA: Choose next action *before* updating Q-value
            next_action = agent.choose_action(next_state)  # Critical for SARSA

            # Update Q-value using (state, action, reward, next_state, next_action)
            agent.update_q_value(state, action, reward, next_state, next_action)  # Pass next_action

            # Move to next state and action
            state = next_state
            action = next_action  # Carry forward the next action

            total_reward += reward
            steps += 1

            
        episode_rewards.append(total_reward)
        episode_steps.append(steps)
        
        if (ep+1) % 100 == 0:
            print(f" Episode {ep+1}: Reward={total_reward}, Steps={steps}")
    
    return episode_rewards, episode_steps

def save_results(complexity, alpha, gamma, epsilon, rewards, steps, passages):
    data = {
        "complexity": complexity,
        "alpha": alpha,
        "gamma": gamma,
        "epsilon": epsilon,
        "extra_passages": passages,
        "avg_rewards": rewards,
        "avg_steps": steps
    }
    
    filename = (f"./results/auto_gen_TD/sarsa_{MAZE_SIZE[0]*2-1}x{MAZE_SIZE[1]*2-1}_comp_{complexity}_"
                f"a_{alpha}_g_{gamma}_e_{epsilon}.json")
    
    with open(filename, 'w') as f:
        json.dump(data, f, indent=2)
    print(f" Saved results to {filename}")

if __name__ == "__main__":
    train_and_evaluate()