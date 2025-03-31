import matplotlib.pyplot as plt
from maze_env.environment import MazeEnv, ComplexMazeEnv
from rl_algorithms.sarsa import SarsaAgent
import numpy as np
import matplotlib.pyplot as plt

# Main Training Loop with Visualization
env = ComplexMazeEnv(maze_file='./generated_mazes/maze_9_9_hard.json') # Create an instance of the Maze environment
state_size = 2
action_size = 4
agent = SarsaAgent(env, action_size=action_size) # Create an instance of the Q-Learning agent, connected to the environment
episodes = 500  # Number of episodes to train the agent for

# Set up a list to store snapshots of Q-tables for visualization at different stages of learning
q_table_snapshots = []
snapshot_intervals = [0, 200, 400, 600, 800, 999]  # Episodes at which to take snapshots of the Q-table

# Training loop
for episode in range(episodes):
    state = env.reset()
    done = False
    total_reward = 0
    steps = 0

    # SARSA requires choosing the first action here
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

    # Record history (same as before)
    agent.rewards_history.append(total_reward)
    agent.steps_history.append(steps)

    # Take snapshots of Q-table at specific intervals for visualization of learning stages
    if episode in snapshot_intervals:
        q_table_snapshots.append((episode, agent.q_table.copy())) # Store a copy of the Q-table along with the episode number

    if episode % 100 == 0: # Print episode completion status every 100 episodes
        print(f"Episode {episode} complete. Total reward: {total_reward}, Steps: {steps}")

# Visualize learning progress (rewards and steps over episodes)
learning_fig = agent.visualize_learning_progress() # Generate the learning progress figure
plt.savefig('results/sarsa_learning_progress.png') # Save the learning progress plot as a PNG file
plt.close(learning_fig) # Close the figure to free up memory

# Visualize the final Q-values and the optimal path based on the learned Q-table
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 7)) # Create a figure with two subplots side-by-side
agent.visualize_q_values(ax1) # Visualize Q-values on the first subplot
optimal_path = agent.get_optimal_path() # Get the optimal path from start to goal based on learned Q-table
agent.visualize_path(optimal_path, ax2) # Visualize the optimal path on the second subplot
plt.savefig('results/sarsa_final_result.png') # Save the final result plot as a PNG file
plt.close(fig) # Close the figure

# Create a 3D heatmap visualization of the Q-values to show the magnitude of Q-values across states
fig = plt.figure(figsize=(15, 10)) # Create a new figure for 3D plot
ax = fig.add_subplot(111, projection='3d') # Add a 3D subplot to the figure

# Get the maximum Q-value for each state (across all actions) to represent the 'value' of each state
max_q_values = np.max(agent.q_table, axis=2) # Find the maximum Q-value for each state across all actions (axis=2 represents actions)

# Create coordinate grids for plotting the 3D surface
x = np.arange(env.size) # x-coordinates (columns)
y = np.arange(env.size) # y-coordinates (rows)
x, y = np.meshgrid(x, y) # Create a meshgrid from x and y coordinates

# Prepare Z-data (Q-values) for the surface plot, masking out walls
z = max_q_values.copy() # Copy the max Q-values
for i in range(env.size): # Iterate through rows
    for j in range(env.size): # Iterate through columns
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
ax.set_xticks(range(env.size)) # Set x-ticks to column numbers
ax.set_yticks(range(env.size)) # Set y-ticks to row numbers

plt.savefig('results/sarsa_q_q_values_3d.png') # Save the 3D Q-value heatmap as a PNG file
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
plt.savefig('results/sarsa_learning_snapshots.png') # Save the learning snapshots plot as a PNG file
plt.show() # Display the learning snapshots plot

# Print the learned Q-table to the console for numerical inspection
print("Learned Q-table:")
print(agent.q_table)
