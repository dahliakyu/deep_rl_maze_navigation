import itertools
import multiprocessing
import time
import numpy as np
import torch
from maze_env.environment import ComplexMazeEnv
from rl_algorithms.pg import PolicyGradientAgent
import matplotlib.pyplot as plt

def train_with_params(params):
    """Training function for parallel execution"""
    lr, gamma, hidden_size, num_episodes = params
    env = ComplexMazeEnv(maze_file='./generated_mazes/maze_5_5_simple.json')
    
    agent = PolicyGradientAgent(
        maze_size=env.size,
        hidden_size=hidden_size,
        lr=lr,
        gamma=gamma
    )
    
    rewards, steps = [], []
    for episode in range(num_episodes):
        state = env.reset()
        log_probs = []
        episode_rewards = []
        done = False
        
        while not done:
            action, log_prob = agent.get_action(state)
            next_state, reward, done = env.step(action)
            log_probs.append(log_prob)
            episode_rewards.append(reward)
            state = next_state
        
        agent.update_policy(log_probs, episode_rewards)
        
        # Store last 10% episodes for evaluation
        if episode >= 0.9 * num_episodes:
            rewards.append(sum(episode_rewards))
            steps.append(len(episode_rewards))
    
    # Return average of last 10% episodes
    return {
        'params': params,
        'avg_reward': np.mean(rewards),
        'avg_steps': np.mean(steps),
        'all_rewards': rewards
    }

def hyperparameter_search():
    # Define search space
    search_space = {
        'learning_rate': [1e-3, 3e-3, 1e-2],
        'gamma': [0.9, 0.95, 0.99],
        'hidden_size': [64, 128, 256],
        'num_episodes': [500, 1000]
    }
    
    # Generate all combinations
    param_combinations = list(itertools.product(
        search_space['learning_rate'],
        search_space['gamma'],
        search_space['hidden_size'],
        search_space['num_episodes']
    ))
    
    # Parallel execution
    num_workers = multiprocessing.cpu_count() - 1
    pool = multiprocessing.Pool(processes=num_workers)
    
    print(f"Starting hyperparameter search with {len(param_combinations)} combinations...")
    start_time = time.time()
    
    results = pool.map(train_with_params, param_combinations)
    
    print(f"Search completed in {time.time()-start_time:.2f} seconds")
    
    # Find best configuration
    best_result = max(results, key=lambda x: x['avg_reward'])
    
    print("\nBest configuration:")
    print(f"Learning rate: {best_result['params'][0]}")
    print(f"Gamma: {best_result['params'][1]}")
    print(f"Hidden size: {best_result['params'][2]}")
    print(f"Episodes: {best_result['params'][3]}")
    print(f"Average reward: {best_result['avg_reward']:.2f}")
    
    # Save results
    torch.save({
        'all_results': results,
        'best_result': best_result
    }, 'hyperparameter_results.pth')
    
    return best_result

def visualize_search_results(results_file):
    data = torch.load(results_file)
    results = data['all_results']
    
    # Plot learning rates vs performance
    plt.figure(figsize=(12, 6))
    for res in results:
        plt.semilogx(res['params'][0], res['avg_reward'], 'bo')
    plt.xlabel('Learning Rate')
    plt.ylabel('Average Reward')
    plt.title('Learning Rate Impact')
    plt.show()
    
    # Similar plots for other parameters
    # ...

if __name__ == "__main__":
    best_config = hyperparameter_search()
    visualize_search_results('hyperparameter_results.pth')