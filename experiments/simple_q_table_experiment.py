import matplotlib.pyplot as plt
from maze_env.environment import MazeEnv, ComplexMazeEnv
from rl_algorithms.simple_q_table import QLearningAgent
import numpy as np
import matplotlib.pyplot as plt

# Main Training Loop with Visualization
env = ComplexMazeEnv(maze_file='genrated_mazes/maze_5_5_mid.json') # Create an instance of the Maze environment
agent = QLearningAgent(env) # Create an instance of the Q-Learning agent, connected to the environment
episodes = 1000  # Number of episodes to train the agent for

# Set up a list to store snapshots of Q-tables for visualization at different stages of learning
q_table_snapshots = []
snapshot_intervals = [0, 200, 400, 600, 800, 999]  # Episodes at which to take snapshots of the Q-table

# Training loop
for episode in range(episodes): # Iterate through each episode
    state = env.reset() # Reset the environment at the start of each episode to the starting state
    done = False # Initialize done flag to False at the start of each episode
    total_reward = 0 # Initialize total reward for the episode to 0
    steps = 0 # Initialize steps for the episode to 0

    while not done: # Run the episode until the goal is reached (done is True)
        action = agent.choose_action(state) # Agent chooses an action based on current state (epsilon-greedy)
        next_state, reward, done = env.step(action) # Take the chosen action in the environment, get next state, reward, and done flag
        agent.update_q_value(state, action, reward, next_state) # Update the Q-value based on the experience (SARSA update rule)
        state = next_state # Update the current state to the next state
        total_reward += reward # Accumulate the reward
        steps += 1 # Increment step count

    # Record history for learning progress visualization
    agent.rewards_history.append(total_reward) # Store the total reward for the episode
    agent.steps_history.append(steps) # Store the number of steps for the episode

    # Take snapshots of Q-table at specific intervals for visualization of learning stages
    if episode in snapshot_intervals:
        q_table_snapshots.append((episode, agent.q_table.copy())) # Store a copy of the Q-table along with the episode number

    if episode % 100 == 0: # Print episode completion status every 100 episodes
        print(f"Episode {episode} complete. Total reward: {total_reward}, Steps: {steps}")

# Visualize learning progress (rewards and steps over episodes)
learning_fig = agent.visualize_learning_progress() # Generate the learning progress figure
plt.savefig('results/simple_q_learning_progress.png') # Save the learning progress plot as a PNG file
plt.close(learning_fig) # Close the figure to free up memory

# Visualize the final Q-values and the optimal path based on the learned Q-table
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 7)) # Create a figure with two subplots side-by-side
agent.visualize_q_values(ax1) # Visualize Q-values on the first subplot
optimal_path = agent.get_optimal_path() # Get the optimal path from start to goal based on learned Q-table
agent.visualize_path(optimal_path, ax2) # Visualize the optimal path on the second subplot
plt.savefig('results/simple_q_final_result.png') # Save the final result plot as a PNG file
plt.close(fig) # Close the figure

# Create a 3D heatmap visualization of the Q-values to show the magnitude of Q-values across states
fig = plt.figure(figsize=(15, 10)) # Create a new figure for 3D plot
ax = fig.add_subplot(111, projection='3d') # Add a 3D subplot to the figure

# Get the maximum Q-value for each state (across all actions) to represent the 'value' of each state
max_q_values = np.max(agent.q_table, axis=2) # Find the maximum Q-value for each state across all actions (axis=2 represents actions)

# Create coordinate grids for plotting the 3D surface
x = np.arange(5) # x-coordinates (columns)
y = np.arange(5) # y-coordinates (rows)
x, y = np.meshgrid(x, y) # Create a meshgrid from x and y coordinates

# Prepare Z-data (Q-values) for the surface plot, masking out walls
z = max_q_values.copy() # Copy the max Q-values
for i in range(5): # Iterate through rows
    for j in range(5): # Iterate through columns
        if env.maze[i, j] == 1:  # If it's a wall
            z[i, j] = 0 # Set Q-value to 0 for walls so they appear at the base of the plot

# Plot the surface using the coordinates and Q-values, using viridis colormap
surf = ax.plot_surface(x, y, z, cmap='viridis', edgecolor='none', alpha=0.8) # Create the 3D surface plot

# Add a colorbar to the 3D plot to show the mapping of color to Q-values
fig.colorbar(surf, ax=ax, shrink=0.5, aspect=5) # Add colorbar to the 3D plot

# Set labels for the axes of the 3D plot
ax.set_xlabel('Y Coordinate') # X-axis label (columns)
ax.set_ylabel('X Coordinate') # Y-axis label (rows)
ax.set_zlabel('Max Q-Value') # Z-axis label (Q-values)
ax.set_title('3D Heatmap of Q-Values') # Set title of the 3D plot

# Set the tick labels to represent grid coordinates
ax.set_xticks(range(15)) # Set x-ticks to column numbers
ax.set_yticks(range(15)) # Set y-ticks to row numbers

plt.savefig('results/simple_q_q_values_3d.png') # Save the 3D Q-value heatmap as a PNG file
plt.close(fig) # Close the figure

# Visualize snapshots of learning at different stages (Q-values at different episodes)
rows = 2 # Number of rows for subplot grid
cols = 3 # Number of columns for subplot grid
fig, axes = plt.subplots(rows, cols, figsize=(18, 10)) # Create a figure with a grid of subplots
axes = axes.flatten() # Flatten the 2D array of axes to make it easier to iterate

for i, (episode, q_table) in enumerate(q_table_snapshots): # Iterate through the snapshots of Q-tables and their corresponding episodes
    # Temporarily override the agent's Q-table with the snapshot for visualization
    original_q_table = agent.q_table.copy() # Store the original Q-table
    agent.q_table = q_table # Temporarily set the agent's Q-table to the snapshot Q-table

    # Visualize Q-values for the current snapshot on a subplot
    agent.visualize_q_values(axes[i]) # Visualize Q-values on the i-th subplot
    axes[i].set_title(f"Episode {episode}") # Set subplot title to the episode number

    # Restore the agent's original Q-table after visualization
    agent.q_table = original_q_table # Restore the original Q-table

plt.tight_layout() # Adjust layout for better subplot spacing
plt.savefig('results/simple_q_learning_snapshots.png') # Save the learning snapshots plot as a PNG file
plt.show() # Display the learning snapshots plot

# Print the learned Q-table to the console for numerical inspection
print("Learned Q-table:")
print(agent.q_table)
