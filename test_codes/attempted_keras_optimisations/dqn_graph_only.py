import matplotlib.pyplot as plt
import numpy as np
import time
import os
import tensorflow as tf
from rl_algorithms.dqn_graph_mode import DQNAgent
from maze_env.environment import ComplexMazeEnv


def train_dqn_agent(env, agent, num_episodes, max_steps=500):
    rewards_history = []
    steps_history = []

    for episode in range(num_episodes):
        state = env.reset()
        total_reward = 0
        steps = 0
        done = False

        while not done and steps < max_steps:
            # Convert state to numpy array for consistent handling
            state_array = np.array(state, dtype=np.float32)
            
            action = agent.act(state_array)
            next_state, reward, done = env.step(action)
            next_state_array = np.array(next_state, dtype=np.float32)
            
            # Store experience for replay
            agent.store_experience(
                state_array, 
                action, 
                reward, 
                next_state_array, 
                done
            )
            
            state = next_state
            
            # Train on batch
            agent.replay()

            total_reward += reward
            steps += 1

        agent.update_epsilon()

        # Print progress
        print(f"Ep {episode+1:4d} | Reward {total_reward:7.2f} | "
              f"Steps {steps:4d} | Eps {agent.epsilon:.3f}")

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


@tf.function
def predict_action(model, state_tensor):
    """TF function for predicting action with highest Q-value"""
    q_values = model(state_tensor)
    return tf.argmax(q_values[0])


def get_optimal_path(env, agent):
    path = []
    state = env.reset()
    done = False
    max_steps = 100  # Increased from 20 to ensure we can find longer paths
    steps = 0
    
    while not done and steps < max_steps:
        path.append(state)
        
        # Create tensor and use graph mode prediction
        state_tensor = tf.convert_to_tensor([state], dtype=tf.float32)
        action = predict_action(agent.q_network, state_tensor).numpy()
        
        next_state, _, done = env.step(action)
        state = next_state
        steps += 1
    
    if not done:
        print("Warning: Could not reach goal within max steps")
    
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


@tf.function
def get_q_values(model, state_tensor):
    """TF function for getting Q-values"""
    return model(state_tensor)


def visualize_dqn_q_values(env, agent):
    fig, ax = plt.subplots(figsize=(10, 10))
    env.render(ax=ax, show_current=False)
    
    # Pre-compute all states for batch prediction
    valid_states = []
    coords = []
    
    for x in range(env.size):
        for y in range(env.size):
            if env.maze[x, y] != 1:  # Not a wall
                valid_states.append([x, y])
                coords.append((x, y))
    
    if not valid_states:
        return fig
    
    # Convert to tensor and predict in batch
    states_tensor = tf.convert_to_tensor(valid_states, dtype=tf.float32)
    q_values_batch = get_q_values(agent.q_network, states_tensor).numpy()
    
    # Process results
    for i, (x, y) in enumerate(coords):
        q_values = q_values_batch[i]
        best_action = np.argmax(q_values)
        q_value = np.max(q_values)
        
        if q_value <= 0:
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
    
    ax.set_title('DQN Q-Values and Optimal Actions')
    return fig


# Main execution
if __name__ == "__main__":
    # Set TensorFlow to work in graph mode where possible
    tf.config.run_functions_eagerly(False)
    
    # Initialize environment and agent
    env = ComplexMazeEnv(maze_file='maze_5_5_mid.json')
    state_size = 2
    action_size = 4
    
    agent = DQNAgent(
        state_size=state_size,
        action_size=action_size,
        learning_rate=0.1,
        gamma=0.99,
        epsilon=1,
        epsilon_decay=0.9995,
        epsilon_min=0.01,
    )
    
    print("Starting DQN training in graph mode...")
    start_time = time.time()
    
    # Train the agent
    rewards, steps = train_dqn_agent(env, agent, num_episodes=500)
    
    print(f"Training completed in {(time.time() - start_time)/60:.2f} minutes")
    
    # Create output directory if it doesn't exist
    os.makedirs('results', exist_ok=True)
    
    # Visualize training progress
    learning_fig = visualize_learning(rewards, steps)
    plt.savefig('results/dqn_learning_progress.png')
    plt.close(learning_fig)
    
    # Visualize optimal path
    optimal_path = get_optimal_path(env, agent)
    path_fig = visualize_path(env, optimal_path)
    plt.savefig('results/dqn_optimal_path.png')
    plt.close(path_fig)
    
    # Visualize Q-values
    q_values_fig = visualize_dqn_q_values(env, agent)
    plt.savefig('results/dqn_q_values.png')
    plt.close(q_values_fig)