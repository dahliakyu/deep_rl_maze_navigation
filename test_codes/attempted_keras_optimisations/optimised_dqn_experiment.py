import matplotlib.pyplot as plt
import numpy as np
import time
from rl_algorithms.optimised_dqn import DQNAgent
from maze_env.environment import ComplexMazeEnv
import os
import tensorflow as tf

def train_dqn_agent(env, agent, num_episodes, max_steps=100):
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
            
            agent.store_experience(
                np.array(state, dtype=np.float32), 
                action, 
                reward, 
                np.array(next_state, dtype=np.float32), 
                done
            )
            
            state = next_state
            total_reward += reward
            steps += 1
            
            # Only replay every few steps for better performance
            if steps % 4 == 0 and len(agent.replay_buffer) >= agent.batch_size:
                agent.replay()

        agent.update_epsilon()

        # Print progress every 10 episodes
        if episode % 10 == 0:
            print(f"Ep {episode+1:4d} | Reward {total_reward:7.2f} | "
                  f"Steps {steps:4d} | Eps {agent.epsilon:.3f}")

        rewards_history.append(total_reward)
        steps_history.append(steps)

    return rewards_history, steps_history

def save_dqn_model(agent, output_dir='models'):
    """Save both Q-network and target network models."""
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Save the main Q-network
    q_network_path = os.path.join(output_dir, 'q_network.keras')
    agent.q_network.save(q_network_path)
    print(f"Q-network saved to {q_network_path}")
    
    # Save the target network
    target_network_path = os.path.join(output_dir, 'target_network.keras')
    agent.target_network.save(target_network_path)
    print(f"Target network saved to {target_network_path}")
    
    # Save agent hyperparameters
    params = {
        'state_size': agent.state_size,
        'action_size': agent.action_size,
        'learning_rate': agent.learning_rate,
        'gamma': agent.gamma,
        'epsilon': agent.epsilon,
        'epsilon_decay': agent.epsilon_decay,
        'epsilon_min': agent.epsilon_min,
        'batch_size': agent.batch_size,
        'target_update_freq': agent.target_update_freq
    }
    
    params_path = os.path.join(output_dir, 'agent_params.json')
    with open(params_path, 'w') as f:
        import json
        json.dump(params, f)
    print(f"Agent parameters saved to {params_path}")
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
    max_steps = 20
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
    
    ax.set_title('Optimal Path Found by DQN')
    return fig

def visualize_dqn_q_values(env, agent):
    fig, ax = plt.subplots(figsize=(10, 10))
    env.render(ax=ax, show_current=False)
    
    for x in range(env.size):
        for y in range(env.size):
            if env.maze[x, y] == 1:
                continue
            
            state = np.array([x, y]).reshape(1, -1)
            q_values = agent.q_network.predict(state, verbose=0)[0]
            best_action = np.argmax(q_values)
            q_value = np.max(q_values)
            
            if q_value <= 0:
                continue
                
            dx, dy = 0, 0
            if best_action == 0:  # Up
                dx = 0
                dy = -0.4
            elif best_action == 1:  # Down
                dx = 0
                dy = 0.4
            elif best_action == 2:  # Left
                dx = -0.4
                dy = 0
            elif best_action == 3:  # Right
                dx = 0.4
                dy = 0
                
            arrow_color = plt.cm.viridis(min(q_value / 10, 1))
            
            ax.arrow(y, x, dy, dx, head_width=0.2, head_length=0.2,
                     fc=arrow_color, ec=arrow_color, width=0.05)
                     
            if q_value > 0:
                ax.text(y + dy * 0.5, x + dx * 0.5, f"{q_value:.1f}",
                        fontsize=8, ha='center', va='center',
                        bbox=dict(facecolor='white', alpha=0.5, pad=1))
    
    ax.set_title('DQN Q-Values and Optimal Actions')
    return fig

# Main execution 
if __name__ == "__main__":
    # Initialize environment and agent
    env = ComplexMazeEnv(maze_file='maze_5_5_simple.json')  # Using 5x5 maze
    state_size = 2
    action_size = 4
    
    # Create agent with optimized parameters for 5x5 maze
    agent = DQNAgent(
        state_size=state_size,
        action_size=action_size,
        learning_rate=0.1,  # Increased for faster convergence
        gamma=0.99,
        epsilon=1.0,
        epsilon_decay=0.9995,  # Faster decay for small maze
        epsilon_min=0.01,
        replay_buffer_size=20000,  # Reduced buffer size
        batch_size=128,  # Smaller batch size for faster iterations
        target_update_freq=100  # More frequent updates
    )

    print("Starting DQN training...")
    start_time = time.time()
    
    # Train the agent with fewer episodes
    rewards, steps = train_dqn_agent(env, agent, num_episodes=500, max_steps=50)
    
    print(f"Training completed in {(time.time() - start_time)/60:.2f} minutes")
    

    os.makedirs('models', exist_ok=True)
    # Save the trained model
    save_dqn_model(agent, output_dir='models/dqn_5_5_mid_maze')

    # Create output directory if it doesn't exist
    os.makedirs('results', exist_ok=True)
    
    # Visualize training progress
    learning_fig = visualize_learning(rewards, steps)
    plt.savefig('results/dqn_learning_progress.png')
    plt.close(learning_fig)
    
    # Visualize optimal path
    optimal_path = get_optimal_path(env, agent)
    path_fig = visualize_path(env, optimal_path)
    plt.savefig('results/dqn_final_result.png')
    plt.close(path_fig)
    
    # Visualize Q-values
    q_values_fig = visualize_dqn_q_values(env, agent)
    plt.savefig('results/dqn_q_values.png')
    plt.close(q_values_fig)