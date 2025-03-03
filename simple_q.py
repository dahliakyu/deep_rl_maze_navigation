import numpy as np
import random
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import matplotlib.patches as patches

# Define the Maze Environment (5x5 grid)
class MazeEnv:
    def __init__(self):
        # Create a 5x5 grid, where 1 represents a wall, 0 is an open space, and 9 is the goal
        self.maze = np.zeros((5, 5))
        self.maze[1, 1] = 1  # Wall
        self.maze[1, 2] = 1  # Wall
        self.maze[1, 3] = 1  # Wall
        self.maze[3, 1] = 1  # Wall
        self.maze[3, 3] = 1  # Wall
        self.maze[4, 4] = 9  # Goal
        
        self.start = (0, 0)  # Starting position
        self.current_state = self.start
        
    def reset(self):
        self.current_state = self.start
        return self.current_state
    
    def step(self, action):
        # Define possible actions: 0 = Up, 1 = Down, 2 = Left, 3 = Right
        x, y = self.current_state
        if action == 0:  # Up
            if x > 0 and self.maze[x - 1, y] != 1:
                x -= 1
        elif action == 1:  # Down
            if x < 4 and self.maze[x + 1, y] != 1:
                x += 1
        elif action == 2:  # Left
            if y > 0 and self.maze[x, y - 1] != 1:
                y -= 1
        elif action == 3:  # Right
            if y < 4 and self.maze[x, y + 1] != 1:
                y += 1
        
        self.current_state = (x, y)
        
        # Check if goal is reached
        if self.maze[x, y] == 9:
            return self.current_state, 10, True  # Reward 10 for reaching the goal
        elif self.maze[x, y] == 1:
            return self.current_state, -1, False  # Penalty for hitting a wall
        else:
            return self.current_state, -0.1, False  # Small penalty for moving

    def render(self, ax=None, show_current=False):
        if ax is None:
            fig, ax = plt.subplots(figsize=(7, 7))
        
        colors = ['white', 'gray', 'green']
        cmap = ListedColormap(colors)
        
        # Create a visualization matrix where 0 is path, 1 is wall, 2 is goal
        vis_maze = self.maze.copy()
        vis_maze[vis_maze == 9] = 2  # Goal is represented as 2 for visualization
        
        ax.imshow(vis_maze, cmap=cmap)
        
        # Plot grid lines
        for i in range(self.maze.shape[0] + 1):
            ax.axhline(i - 0.5, color='black', linewidth=1)
            ax.axvline(i - 0.5, color='black', linewidth=1)
        
        # Mark start position
        start_y, start_x = self.start[0], self.start[1]
        ax.text(start_x, start_y, 'S', ha='center', va='center', fontsize=20, color='blue')
        
        # Mark goal position
        goal_idx = np.where(self.maze == 9)
        if len(goal_idx[0]) > 0:
            goal_y, goal_x = goal_idx[0][0], goal_idx[1][0]
            ax.text(goal_x, goal_y, 'G', ha='center', va='center', fontsize=20, color='white')
        
        # Show current position if requested
        if show_current:
            curr_y, curr_x = self.current_state
            ax.add_patch(patches.Circle((curr_x, curr_y), 0.3, color='red'))
        
        # Label the axes with coordinates
        ax.set_xticks(range(self.maze.shape[1]))
        ax.set_yticks(range(self.maze.shape[0]))
        ax.grid(False)
        
        return ax

# Q-Learning Agent
class QLearningAgent:
    def __init__(self, env, alpha=0.1, gamma=0.9, epsilon=0.1):
        self.env = env
        self.alpha = alpha  # Learning rate
        self.gamma = gamma  # Discount factor
        self.epsilon = epsilon  # Exploration factor
        self.q_table = np.zeros((5, 5, 4))  # Q-table for each state-action pair (5x5 grid, 4 actions)
        self.rewards_history = []  # To track rewards over episodes
        self.steps_history = []    # To track steps over episodes
    
    def choose_action(self, state):
        if random.uniform(0, 1) < self.epsilon:
            return random.randint(0, 3)  # Explore
        else:
            x, y = state
            return np.argmax(self.q_table[x, y])  # Exploit
    
    def update_q_value(self, state, action, reward, next_state):
        x, y = state
        next_x, next_y = next_state
        max_q_next = np.max(self.q_table[next_x, next_y])  # Max Q-value for next state
        self.q_table[x, y, action] = self.q_table[x, y, action] + self.alpha * (reward + self.gamma * max_q_next - self.q_table[x, y, action])
    
    def get_optimal_path(self):
        """Generate the optimal path from start to goal based on current Q-table"""
        path = []
        state = self.env.reset()
        done = False
        max_steps = 25  # Prevent infinite loops
        step_count = 0
        
        while not done and step_count < max_steps:
            path.append(state)
            x, y = state
            action = np.argmax(self.q_table[x, y])  # Choose best action
            next_state, _, done = self.env.step(action)
            state = next_state
            step_count += 1
        
        path.append(state)  # Add final state
        return path

    def visualize_q_values(self, ax=None):
        """Visualize the Q-values for each state"""
        if ax is None:
            fig, ax = plt.subplots(figsize=(10, 10))
        
        # Render the base maze
        self.env.render(ax)
        
        # Add arrows to show optimal action in each state
        for i in range(5):
            for j in range(5):
                if self.env.maze[i, j] == 1:  # Skip walls
                    continue
                
                # Get best action
                best_action = np.argmax(self.q_table[i, j])
                q_value = np.max(self.q_table[i, j])
                
                # Skip if no clear preference yet
                if q_value <= 0:
                    continue
                
                # Draw an arrow pointing in the direction of the best action
                dx, dy = 0, 0
                if best_action == 0:  # Up
                    dx, dy = 0, -0.4
                elif best_action == 1:  # Down
                    dx, dy = 0, 0.4
                elif best_action == 2:  # Left
                    dx, dy = -0.4, 0
                elif best_action == 3:  # Right
                    dx, dy = 0.4, 0
                
                # Arrow color based on Q-value strength
                arrow_color = plt.cm.viridis(min(q_value / 10, 1))
                
                # Fixed: Properly center the arrows in each cell
                ax.arrow(j, i, dx, dy, head_width=0.2, head_length=0.2, fc=arrow_color, ec=arrow_color, width=0.05)
                
                # Optionally print Q-value (offset to not overlap with arrow)
                if q_value > 0:
                    # Position the Q-value text to avoid overlapping with the arrow
                    text_offset_x = -0.2 if dx > 0 else 0.2 if dx < 0 else 0
                    text_offset_y = -0.2 if dy > 0 else 0.2 if dy < 0 else 0
                    ax.text(j + text_offset_x, i + text_offset_y, f"{q_value:.1f}", 
                           fontsize=8, ha='center', va='center', 
                           bbox=dict(facecolor='white', alpha=0.5, pad=1))
        
        ax.set_title("Q-values and Optimal Actions")
        return ax

    def visualize_path(self, path, ax=None):
        """Visualize a path through the maze"""
        if ax is None:
            fig, ax = plt.subplots(figsize=(7, 7))
        
        # Render the base maze
        self.env.render(ax)
        
        # Plot the path
        path_x = [p[1] for p in path]
        path_y = [p[0] for p in path]
        ax.plot(path_x, path_y, 'o-', color='blue', markersize=10, alpha=0.6)
        
        # Number the steps
        for i, (y, x) in enumerate(zip(path_y, path_x)):
            ax.text(x, y, str(i), fontsize=12, ha='center', va='center', color='red')
        
        ax.set_title("Optimal Path")
        return ax

    def visualize_learning_progress(self):
        """Visualize the learning progress over episodes"""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
        
        # Plot rewards
        window_size = min(50, len(self.rewards_history))
        if window_size > 0:
            rolling_avg = np.convolve(self.rewards_history, np.ones(window_size)/window_size, mode='valid')
            ax1.plot(self.rewards_history, alpha=0.3, color='blue', label='Episode Reward')
            ax1.plot(range(window_size-1, len(self.rewards_history)), rolling_avg, color='red', label=f'{window_size}-Episode Average')
            ax1.set_xlabel('Episode')
            ax1.set_ylabel('Total Reward')
            ax1.set_title('Rewards over Episodes')
            ax1.legend()
        
        # Plot steps
        if len(self.steps_history) > 0:
            window_size = min(50, len(self.steps_history))
            rolling_avg = np.convolve(self.steps_history, np.ones(window_size)/window_size, mode='valid')
            ax2.plot(self.steps_history, alpha=0.3, color='green', label='Steps per Episode')
            ax2.plot(range(window_size-1, len(self.steps_history)), rolling_avg, color='red', label=f'{window_size}-Episode Average')
            ax2.set_xlabel('Episode')
            ax2.set_ylabel('Steps')
            ax2.set_title('Steps per Episode')
            ax2.legend()
        
        plt.tight_layout()
        return fig

# Main Training Loop with Visualization
env = MazeEnv()
agent = QLearningAgent(env)
episodes = 1000  # Number of episodes for training

# Set up a list to store snapshots for visualization
q_table_snapshots = []
snapshot_intervals = [0, 10, 50, 100, 500, 999]  # Episodes to take snapshots at

# Training loop
for episode in range(episodes):
    state = env.reset()
    done = False
    total_reward = 0
    steps = 0
    
    while not done:
        action = agent.choose_action(state)
        next_state, reward, done = env.step(action)
        agent.update_q_value(state, action, reward, next_state)
        state = next_state
        total_reward += reward
        steps += 1
    
    # Record history
    agent.rewards_history.append(total_reward)
    agent.steps_history.append(steps)
    
    # Take snapshots at specific intervals
    if episode in snapshot_intervals:
        q_table_snapshots.append((episode, agent.q_table.copy()))
    
    if episode % 100 == 0:
        print(f"Episode {episode} complete. Total reward: {total_reward}, Steps: {steps}")

# Visualize learning progress
learning_fig = agent.visualize_learning_progress()
plt.savefig('simple_q_results/learning_progress.png')
plt.close(learning_fig)

# Visualize the final Q-values and optimal path
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 7))
agent.visualize_q_values(ax1)
optimal_path = agent.get_optimal_path()
agent.visualize_path(optimal_path, ax2)
plt.savefig('simple_q_results/final_result.png')
plt.close(fig)

# Create a 3D heatmap of the Q-values
fig = plt.figure(figsize=(15, 10))
ax = fig.add_subplot(111, projection='3d')

# Get the max Q-value for each state
max_q_values = np.max(agent.q_table, axis=2)

# Create x and y coordinates
x = np.arange(5)
y = np.arange(5)
x, y = np.meshgrid(x, y)

# Remove walls from visualization
z = max_q_values.copy()
for i in range(5):
    for j in range(5):
        if env.maze[i, j] == 1:  # Wall
            z[i, j] = 0

# Plot the surface
surf = ax.plot_surface(x, y, z, cmap='viridis', edgecolor='none', alpha=0.8)

# Add colorbar
fig.colorbar(surf, ax=ax, shrink=0.5, aspect=5)

# Set labels
ax.set_xlabel('Y Coordinate')
ax.set_ylabel('X Coordinate')
ax.set_zlabel('Max Q-Value')
ax.set_title('3D Heatmap of Q-Values')

# Set the tick labels
ax.set_xticks(range(5))
ax.set_yticks(range(5))

plt.savefig('simple_q_results/q_values_3d.png')
plt.close(fig)

# Visualize snapshots of learning at different stages
rows = 2
cols = 3
fig, axes = plt.subplots(rows, cols, figsize=(18, 10))
axes = axes.flatten()

for i, (episode, q_table) in enumerate(q_table_snapshots):
    # Temporarily override the Q-table for visualization
    original_q_table = agent.q_table.copy()
    agent.q_table = q_table
    
    # Visualize
    agent.visualize_q_values(axes[i])
    axes[i].set_title(f"Episode {episode}")
    
    # Restore original Q-table
    agent.q_table = original_q_table

plt.tight_layout()
plt.savefig('simple_q_results/learning_snapshots.png')
plt.show()

# Print the learned Q-table
print("Learned Q-table:")
print(agent.q_table)