import matplotlib.pyplot as plt
import numpy as np
import time
import torch
from maze_env.environment import ComplexMazeEnv
import os
from rl_algorithms.pg import PolicyGradientAgent  

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def train_pg_agent(env, agent, num_episodes):
    rewards_history = []
    steps_history = []
    max_steps = 500

    for episode in range(num_episodes):
        state = env.reset()
        episode_log_probs = []
        episode_rewards = []
        done = False
        steps = 0

        while not done and steps < max_steps:
            action, log_prob = agent.get_action(state)
            next_state, reward, done = env.step(action)
            
            episode_log_probs.append(log_prob)
            episode_rewards.append(reward)
            
            state = next_state
            steps += 1

        # Update policy with the episode's data
        agent.update_policy(episode_log_probs, episode_rewards)
        
        total_reward = sum(episode_rewards)
        rewards_history.append(total_reward)
        steps_history.append(steps)

        if episode % 50 == 0:
            print(f"Episode {episode} complete. Total reward: {total_reward}, Steps: {steps}")

    return rewards_history, steps_history

def visualize_learning(rewards_history, steps_history):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
    
    ax1.plot(rewards_history, label='Reward per Episode')
    ax1.set_xlabel('Episode')
    ax1.set_ylabel('Total Reward')
    ax1.set_title('Training Rewards')
    
    ax2.plot(steps_history, label='Steps per Episode')
    ax2.set_xlabel('Episode')
    ax2.set_ylabel('Steps')
    ax2.set_title('Training Steps')
    
    plt.tight_layout()
    return fig

def get_optimal_path(env, agent):
    path = []
    state = env.reset()
    done = False
    max_steps = 20
    steps = 0
    
    while not done and steps < max_steps:
        path.append(state)
        action, _ = agent.get_action(state)  # Stochastic policy
        next_state, _, done = env.step(action)
        state = next_state
        steps += 1
    
    path.append(state)
    return path

def visualize_path(env, path):
    fig, ax = plt.subplots(figsize=(10, 10))
    env.render(ax=ax, show_current=False)
    
    path_y = [p[1] for p in path]
    path_x = [p[0] for p in path]
    
    ax.plot(path_y, path_x, 'o-', color='blue', markersize=10, alpha=0.6)
    
    for i, (x, y) in enumerate(path):
        ax.text(y, x, str(i), ha='center', va='center', color='red', fontsize=12)
    
    ax.set_title('Path Found by Policy Gradient')
    return fig

def visualize_policy(env, agent):
    fig, ax = plt.subplots(figsize=(10, 10))
    env.render(ax=ax, show_current=False)
    
    for x in range(env.size):
        for y in range(env.size):
            if env.maze[x, y] == 1:
                continue
            
            state = (x, y)
            normalized_state = torch.FloatTensor([
                x/(env.size-1), 
                y/(env.size-1)
            ]).unsqueeze(0)
            
            with torch.no_grad():
                probs = agent.policy_network(normalized_state).numpy()[0]
                
            best_action = np.argmax(probs)
            action_prob = np.max(probs)
            
            dx, dy = 0, 0
            if best_action == 0:  # Up
                dy = -0.4
            elif best_action == 1:  # Down
                dy = 0.4
            elif best_action == 2:  # Left
                dx = -0.4
            elif best_action == 3:  # Right
                dx = 0.4
                
            arrow_color = plt.cm.viridis(min(action_prob / 1.0, 1))
            
            ax.arrow(y, x, dy, dx, head_width=0.2, head_length=0.2,
                     fc=arrow_color, ec=arrow_color, width=0.05)
                     
            ax.text(y + dy * 0.5, x + dx * 0.5, f"{action_prob:.2f}",
                    fontsize=8, ha='center', va='center',
                    bbox=dict(facecolor='white', alpha=0.5, pad=1))
    
    ax.set_title('Policy Gradient Action Probabilities')
    return fig

if __name__ == "__main__":
    env = ComplexMazeEnv(maze_file='./generated_mazes/maze_5_5_simple.json')
    
    agent = PolicyGradientAgent(
        maze_size=env.size,
        hidden_size=128,
        lr=0.01,
        gamma=0.99
    )
    
    print("Starting Policy Gradient training...")
    start_time = time.time()
    
    rewards, steps = train_pg_agent(env, agent, num_episodes=1000)
    
    print(f"Training completed in {(time.time() - start_time)/60:.2f} minutes")
    
    os.makedirs('results', exist_ok=True)
    
    # Save learning curves
    learning_fig = visualize_learning(rewards, steps)
    plt.savefig('results/pg_learning_progress.png')
    plt.close(learning_fig)
    
    # Save path visualization
    optimal_path = get_optimal_path(env, agent)
    path_fig = visualize_path(env, optimal_path)
    plt.savefig('results/pg_final_result.png')
    plt.close(path_fig)
    
    # Save policy visualization
    policy_fig = visualize_policy(env, agent)
    plt.savefig('results/pg_policy_visualization.png')
    plt.close(policy_fig)