import matplotlib.pyplot as plt
import numpy as np
import time
import os
import torch
from maze_env.environment import ComplexMazeEnv
from rl_algorithms.ddqn_torch import DDQNAgent

def train_ddqn_agent(env, agent, num_episodes, max_steps=2000):
    rewards_history = []
    steps_history = []

    for episode in range(num_episodes):
        state = env.reset()
        total_reward = 0
        steps = 0
        done = False

        while not done and steps < max_steps:
            action = agent.act(state)
            next_state, reward, done = env.step(action)
            
            # Store experience in replay buffer
            agent.store_experience(
                np.array(state), 
                action, 
                reward, 
                np.array(next_state), 
                done
            )
            
            state = next_state
            agent.replay()  # Train on batch from replay buffer

            total_reward += reward
            steps += 1

        # Update epsilon after each episode
        agent.update_epsilon()

        # Print progress
        if episode % 50 == 0:
            print(f"Episode {episode}/{num_episodes} - Reward: {total_reward:.2f}, Steps: {steps}, Epsilon: {agent.epsilon:.4f}")

        rewards_history.append(total_reward)
        steps_history.append(steps)

    return rewards_history, steps_history

def visualize_learning(rewards_history, steps_history):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
    
    # Plot rewards
    ax1.plot(rewards_history, label='Reward per Episode')
    ax1.set_xlabel('Episode')
    ax1.set_ylabel('Total Reward')
    ax1.set_title('Training Rewards')
    
    # Plot steps
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
    max_steps = 100
    steps = 0
    
    while not done and steps < max_steps:
        path.append(state)
        action = agent.act(state)
        next_state, _, done = env.step(action)
        state = next_state
        steps += 1
    
    path.append(state)
    return path

def visualize_path(env, path):
    fig, ax = plt.subplots(figsize=(10, 10))
    env.render(ax=ax, show_current=False)
    
    # Correctly handle the path coordinates
    path_y = [p[1] for p in path]  # y coordinate is the second element
    path_x = [p[0] for p in path]  # x coordinate is the first element
    
    ax.plot(path_y, path_x, 'o-', color='blue', markersize=10, alpha=0.6)
    
    for i, (x, y) in enumerate(path):
        ax.text(y, x, str(i), ha='center', va='center', color='red', fontsize=12)
    
    ax.set_title('Optimal Path Found by DDQN')
    return fig

def visualize_ddqn_q_values(env, agent):
    fig, ax = plt.subplots(figsize=(10, 10))
    env.render(ax=ax, show_current=False)
    
    device = agent.device
    
    for x in range(env.size):
        for y in range(env.size):
            if env.maze[x, y] == 1:  # Skip walls
                continue
            
            state = torch.FloatTensor([x, y]).unsqueeze(0).to(device)
            agent.q_network.eval()
            with torch.no_grad():
                q_values = agent.q_network(state).cpu().numpy()[0]
            agent.q_network.train()
            
            best_action = np.argmax(q_values)
            q_value = np.max(q_values)
            
            if q_value <= 0:  # Skip drawing arrows for non-positive Q-values
                continue
                
            dx, dy = 0, 0
            if best_action == 0:  # Up
                dx = -0.4
                dy = 0
            elif best_action == 1:  # Down
                dx = 0.4
                dy = 0
            elif best_action == 2:  # Left
                dx = 0
                dy = -0.4
            elif best_action == 3:  # Right
                dx = 0
                dy = 0.4
                
            arrow_color = plt.cm.viridis(min(q_value / 10, 1))
            
            ax.arrow(y, x, dy, dx, head_width=0.2, head_length=0.2,
                     fc=arrow_color, ec=arrow_color, width=0.05)
                     
            if q_value > 0:
                ax.text(y + dy * 0.5, x + dx * 0.5, f"{q_value:.1f}",
                        fontsize=8, ha='center', va='center',
                        bbox=dict(facecolor='white', alpha=0.5, pad=1))
    
    ax.set_title('DDQN Q-Values and Optimal Actions')
    return fig

# Main execution
if __name__ == "__main__":
    # Create output directories if they don't exist
    os.makedirs('experiments/results', exist_ok=True)
    
    # Initialize environment and agent
    env = ComplexMazeEnv(maze_config='./manually_drawn_mazes/maze_9_9.json')
    state_size = 2  # (x, y) coordinates
    action_size = 4  # Up, Down, Left, Right
    
    agent = DDQNAgent(
        state_size=state_size,
        action_size=action_size,
        learning_rate=0.001,
        gamma=0.99,
        epsilon=1.0,
        epsilon_decay=0.995,
        epsilon_min=0.01,
        replay_buffer_size=10000,
        batch_size=64,
        target_update_freq=10
    )
    
    print("Starting DDQN training...")
    start_time = time.time()

    # Train the agent
    num_episodes = 500
    rewards, steps = train_ddqn_agent(env, agent, num_episodes=num_episodes)

    print(f"Training completed in {(time.time() - start_time)/60:.2f} minutes")

    # Visualize training progress
    try:
        learning_fig = visualize_learning(rewards, steps)
        learning_fig.savefig('experiments/results/ddqn_learning.png')
        plt.close(learning_fig)
        print("Saved learning progress plot.")
    except Exception as e:
        print(f"Error saving learning progress: {e}")

    # Visualize optimal path
    try:
        optimal_path = get_optimal_path(env, agent)
        path_fig = visualize_path(env, optimal_path)
        path_fig.savefig('experiments/results/ddqn_path.png')
        plt.close(path_fig)
        print("Saved optimal path plot.")
    except Exception as e:
        print(f"Error saving optimal path: {e}")

    # Visualize Q-values
    try:
        q_values_fig = visualize_ddqn_q_values(env, agent)
        q_values_fig.savefig('experiments/results/ddqn_q_values.png')
        plt.close(q_values_fig)
        print("Saved Q-values plot.")
    except Exception as e:
        print(f"Error saving Q-values: {e}")

    print("Training and visualization completed successfully!")