import matplotlib.pyplot as plt
import numpy as np
import time
from rl_algorithms.dqn import DQNAgent
from maze_env.environment import ComplexMazeEnv
import os


def train_dqn_agent(env, agent, num_episodes, update_target_every=10):
    rewards_history = []
    steps_history = []
    max_steps = 500

    for episode in range(num_episodes):
        state = env.reset()
        total_reward = 0
        steps = 0
        done = False

        while not done and steps < max_steps:
            action = agent.act(state)
            next_state, reward, done = env.step(action)
            
            # Convert states to numpy arrays for neural network compatibility
            agent.store_experience(
                np.array(state), 
                action, 
                reward, 
                np.array(next_state), 
                done
            )
            state = next_state
            agent.replay()

            total_reward += reward
            steps += 1

        if episode % update_target_every == 0:
            agent.update_target_network()

        agent.update_epsilon()

        # Always show progress
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
    env = ComplexMazeEnv(maze_file='./generated_mazes/maze_5_5_simple.json')
    state_size = 2
    action_size = 4
    
    agent = DQNAgent(
        state_size=state_size,
        action_size=action_size,
        learning_rate=0.1,    # Increased learning rate
        gamma=0.99,            # Reduced gamma for shorter-term rewards
        epsilon=1,
        epsilon_decay=0.9995,    # Faster exploration decay
        epsilon_min=0.01,      # Higher minimum exploration
    )
    
    print("Starting DQN training...")
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
    plt.savefig('results/dqn_final_result.png')
    plt.close(path_fig)
    
    # Visualize Q-values
    q_values_fig = visualize_dqn_q_values(env, agent)
    plt.savefig('results/dqn_q_values.png')
    plt.close(q_values_fig)
