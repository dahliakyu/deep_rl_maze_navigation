import numpy as np
import random
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import matplotlib.patches as patches
from mpl_toolkits.mplot3d import Axes3D  # Import for 3D plotting

# Define the Maze Environment (5x5 grid)
class MazeEnv:
    def __init__(self):
        """
        Initializes the Maze environment.

        This sets up a 5x5 grid representing the maze.
        - 0 represents an open space (path).
        - 1 represents a wall (obstacle).
        - 9 represents the goal.

        It also defines the starting position and sets the current state to the start.
        """
        # Create a 5x5 grid, initialized with zeros (open spaces)
        self.maze = np.zeros((5, 5))
        # Define walls in the maze
        self.maze[1, 1] = 1  # Wall at row 1, column 1
        self.maze[1, 2] = 1  # Wall at row 1, column 2
        self.maze[1, 3] = 1  # Wall at row 1, column 3
        self.maze[3, 1] = 1  # Wall at row 3, column 1
        self.maze[3, 3] = 1  # Wall at row 3, column 3
        # Define the goal position
        self.maze[4, 4] = 9  # Goal at row 4, column 4

        self.start = (0, 0)  # Starting position is at the top-left corner (row 0, column 0)
        self.current_state = self.start # Initialize the current state to the starting position

    def reset(self):
        """
        Resets the environment to the starting state.

        This method is called at the beginning of each episode to start from the initial position.
        It sets the current state back to the starting position and returns the starting state.
        """
        self.current_state = self.start # Set the current state back to the starting position
        return self.current_state # Return the starting state

    def step(self, action):
        """
        Executes an action in the environment and returns the next state, reward, and done flag.

        Actions are defined as:
        - 0: Up
        - 1: Down
        - 2: Left
        - 3: Right

        The method updates the agent's position based on the action, considering boundaries and walls.
        It calculates the reward based on the outcome of the action:
        - +10 for reaching the goal.
        - -1 for hitting a wall (though the position doesn't change when hitting a wall).
        - -0.1 for moving to an open space (step penalty).
        - done is True if the goal is reached, False otherwise.

        Args:
            action (int): The action to take (0, 1, 2, or 3).

        Returns:
            tuple: (next_state, reward, done)
                   - next_state (tuple): The state after taking the action (row, column).
                   - reward (float): The reward obtained after taking the action.
                   - done (bool): True if the episode ends (goal reached), False otherwise.
        """
        x, y = self.current_state # Get the current row and column from the current state
        if action == 0:  # Up
            if x > 0 and self.maze[x - 1, y] != 1: # Check if moving up is within bounds and not a wall
                x -= 1 # Move up if possible
        elif action == 1:  # Down
            if x < 4 and self.maze[x + 1, y] != 1: # Check if moving down is within bounds and not a wall
                x += 1 # Move down if possible
        elif action == 2:  # Left
            if y > 0 and self.maze[x, y - 1] != 1: # Check if moving left is within bounds and not a wall
                y -= 1 # Move left if possible
        elif action == 3:  # Right
            if y < 4 and self.maze[x, y + 1] != 1: # Check if moving right is within bounds and not a wall
                y += 1 # Move right if possible

        self.current_state = (x, y) # Update the current state to the new position

        # Check for rewards and termination conditions
        if self.maze[x, y] == 9: # If the agent reached the goal
            return self.current_state, 10, True  # Reward 10 for reaching the goal, episode is done
        elif self.maze[x, y] == 1: # If the agent moved into a wall (this condition is actually never met as the agent doesn't move into a wall)
            return self.current_state, -1, False  # Penalty for hitting a wall (though position doesn't change)
        else: # If the agent moved to an open space
            return self.current_state, -0.1, False  # Small penalty for moving (to encourage shorter paths)

    def render(self, ax=None, show_current=False):
        """
        Renders the maze environment visually using matplotlib.

        It displays the maze grid, walls, goal, start position, and optionally the current agent position.

        Args:
            ax (matplotlib.axes._axes.Axes, optional): An existing matplotlib Axes object to draw on.
                                                      If None, a new figure and axes are created. Defaults to None.
            show_current (bool, optional): If True, shows the current position of the agent as a red circle. Defaults to False.

        Returns:
            matplotlib.axes._axes.Axes: The Axes object on which the maze is rendered.
        """
        if ax is None: # If no Axes object is provided, create a new figure and Axes
            fig, ax = plt.subplots(figsize=(7, 7))

        colors = ['white', 'gray', 'green'] # Define colors for path, wall, and goal
        cmap = ListedColormap(colors) # Create a colormap from the list of colors

        # Create a visualization matrix: 0 for path, 1 for wall, 2 for goal
        vis_maze = self.maze.copy() # Create a copy of the maze to avoid modifying the original
        vis_maze[vis_maze == 9] = 2  # Represent goal (9) as 2 for visualization

        ax.imshow(vis_maze, cmap=cmap) # Display the maze as an image using the defined colormap

        # Plot grid lines to clearly separate cells
        for i in range(self.maze.shape[0] + 1): # Iterate through rows and columns + 1 to draw all grid lines
            ax.axhline(i - 0.5, color='black', linewidth=1) # Horizontal grid lines
            ax.axvline(i - 0.5, color='black', linewidth=1) # Vertical grid lines

        # Mark start position with 'S'
        start_y, start_x = self.start[0], self.start[1] # Get start row and column
        ax.text(start_x, start_y, 'S', ha='center', va='center', fontsize=20, color='blue') # Add 'S' text at start position

        # Mark goal position with 'G'
        goal_idx = np.where(self.maze == 9) # Find the index (row, column) of the goal
        if len(goal_idx[0]) > 0: # Check if a goal exists (should always be true in this maze)
            goal_y, goal_x = goal_idx[0][0], goal_idx[1][0] # Get goal row and column
            ax.text(goal_x, goal_y, 'G', ha='center', va='center', fontsize=20, color='white') # Add 'G' text at goal position

        # Show current position if requested
        if show_current: # If show_current is True
            curr_y, curr_x = self.current_state # Get current row and column
            ax.add_patch(patches.Circle((curr_x, curr_y), 0.3, color='red')) # Add a red circle to represent current position

        # Label the axes with coordinates (optional, for clarity)
        ax.set_xticks(range(self.maze.shape[1])) # Set x-axis ticks to column numbers
        ax.set_yticks(range(self.maze.shape[0])) # Set y-axis ticks to row numbers
        ax.grid(False) # Turn off default grid lines, as we've drawn our own

        return ax # Return the Axes object for further modifications if needed

# Q-Learning Agent
class QLearningAgent:
    def __init__(self, env, alpha=0.1, gamma=0.9, epsilon=0.1):
        """
        Initializes the Q-Learning Agent.

        Args:
            env (MazeEnv): The maze environment the agent will interact with.
            alpha (float, optional): Learning rate, determines how much new information overrides old information. Defaults to 0.1.
            gamma (float, optional): Discount factor, determines the importance of future rewards. Defaults to 0.9.
            epsilon (float, optional): Exploration rate, probability of choosing a random action instead of exploitation. Defaults to 0.1.
        """
        self.env = env # Store the environment
        self.alpha = alpha  # Learning rate
        self.gamma = gamma  # Discount factor
        self.epsilon = epsilon  # Exploration factor
        self.q_table = np.zeros((5, 5, 4))  # Q-table initialized to zeros. Dimensions: (rows, columns, actions)
        # self.q_table[state, action] stores the Q-value for taking 'action' in 'state'
        self.rewards_history = []  # List to store total rewards for each episode
        self.steps_history = []    # List to store number of steps taken in each episode

    def choose_action(self, state):
        """
        Chooses an action based on the current state using an epsilon-greedy policy.

        With probability epsilon, it explores (chooses a random action).
        With probability 1-epsilon, it exploits (chooses the action with the highest Q-value in the current state).

        Args:
            state (tuple): The current state (row, column).

        Returns:
            int: The chosen action (0, 1, 2, or 3).
        """
        if random.uniform(0, 1) < self.epsilon: # Generate a random number between 0 and 1, explore if less than epsilon
            return random.randint(0, 3)  # Explore: choose a random action (0, 1, 2, 3)
        else:
            x, y = state # Get row and column from the state
            return np.argmax(self.q_table[x, y])  # Exploit: choose the action with the highest Q-value for the current state

    def update_q_value(self, state, action, reward, next_state):
        """
        Updates the Q-value for a given state-action pair using the Q-learning update rule.

        Q(s, a) = Q(s, a) + alpha * [reward + gamma * max_a' Q(s', a') - Q(s, a)]
        where:
        - s is the current state
        - a is the action taken
        - reward is the reward received
        - s' is the next state
        - a' are all possible actions in the next state
        - alpha is the learning rate
        - gamma is the discount factor

        Args:
            state (tuple): The current state (row, column).
            action (int): The action taken (0, 1, 2, or 3).
            reward (float): The reward received after taking the action.
            next_state (tuple): The next state (row, column) after taking the action.
        """
        x, y = state # Get row and column of the current state
        next_x, next_y = next_state # Get row and column of the next state
        max_q_next = np.max(self.q_table[next_x, next_y])  # Find the maximum Q-value among all possible actions in the next state (best estimated future reward)
        # Q-learning update rule:
        self.q_table[x, y, action] = self.q_table[x, y, action] + self.alpha * (reward + self.gamma * max_q_next - self.q_table[x, y, action])
        # Q(s, a) = Q(s, a) + alpha * (target - current_Q)
        # target = reward + gamma * max_a' Q(s', a')

    def get_optimal_path(self):
        """
        Generates the optimal path from the start to the goal based on the learned Q-table.

        It starts from the initial state and iteratively chooses the action with the highest Q-value until the goal is reached or a maximum number of steps is exceeded (to prevent infinite loops).

        Returns:
            list: A list of states (tuples of (row, column)) representing the optimal path.
        """
        path = [] # Initialize an empty list to store the path
        state = self.env.reset() # Reset the environment to the starting state
        done = False # Initialize done flag to False
        max_steps = 25  # Maximum steps to prevent infinite loops if no path is found or agent gets stuck
        step_count = 0 # Initialize step counter

        while not done and step_count < max_steps: # Loop until goal is reached or max steps exceeded
            path.append(state) # Add the current state to the path
            x, y = state # Get row and column of the current state
            action = np.argmax(self.q_table[x, y])  # Choose the action with the highest Q-value (exploitation)
            next_state, _, done = self.env.step(action) # Take the chosen action in the environment
            state = next_state # Update the current state to the next state
            step_count += 1 # Increment step counter

        path.append(state)  # Add the final state to the path
        return path # Return the list of states representing the optimal path

    def visualize_q_values(self, ax=None):
        """
        Visualizes the Q-values for each state in the maze using arrows to indicate optimal actions and color intensity to represent Q-value magnitude.

        Args:
            ax (matplotlib.axes._axes.Axes, optional): An existing matplotlib Axes object to draw on.
                                                      If None, a new figure and axes are created. Defaults to None.

        Returns:
            matplotlib.axes._axes.Axes: The Axes object on which the Q-values are visualized.
        """
        if ax is None: # If no Axes object is provided, create a new figure and Axes
            fig, ax = plt.subplots(figsize=(10, 10))

        # Render the base maze to show walls and goal
        self.env.render(ax)

        # Add arrows to show optimal action in each state
        for i in range(5): # Iterate through rows
            for j in range(5): # Iterate through columns
                if self.env.maze[i, j] == 1:  # Skip walls, no action needed in wall states
                    continue

                # Get best action and its Q-value for the current state (i, j)
                best_action = np.argmax(self.q_table[i, j])
                q_value = np.max(self.q_table[i, j])

                # Skip if no clear preference yet (Q-value is close to zero or negative, meaning not learned well yet)
                if q_value <= 0:
                    continue

                # Determine arrow direction based on the best action
                dx, dy = 0, 0 # Initialize arrow direction to no movement
                if best_action == 0:  # Up
                    dx, dy = 0, -0.4 # Arrow pointing up
                elif best_action == 1:  # Down
                    dx, dy = 0, 0.4 # Arrow pointing down
                elif best_action == 2:  # Left
                    dx, dy = -0.4, 0 # Arrow pointing left
                elif best_action == 3:  # Right
                    dx, dy = 0.4, 0 # Arrow pointing right

                # Arrow color based on Q-value strength, using viridis colormap
                arrow_color = plt.cm.viridis(min(q_value / 10, 1)) # Normalize Q-value to range [0, 1] for color mapping, cap at 10 for visualization

                # Properly center the arrows in each cell
                ax.arrow(j, i, dx, dy, head_width=0.2, head_length=0.2, fc=arrow_color, ec=arrow_color, width=0.05) # Draw arrow on the plot

                # Optionally print Q-value near the arrow
                if q_value > 0:
                    # Position the Q-value text to avoid overlapping with the arrow
                    text_offset_x = -0.2 if dx > 0 else 0.2 if dx < 0 else 0 # Offset text horizontally if arrow is horizontal
                    text_offset_y = -0.2 if dy > 0 else 0.2 if dy < 0 else 0 # Offset text vertically if arrow is vertical
                    ax.text(j + text_offset_x, i + text_offset_y, f"{q_value:.1f}", # Add Q-value text
                           fontsize=8, ha='center', va='center',
                           bbox=dict(facecolor='white', alpha=0.5, pad=1)) # Add a white box around the text for better readability

        ax.set_title("Q-values and Optimal Actions") # Set plot title
        return ax # Return the Axes object

    def visualize_path(self, path, ax=None):
        """
        Visualizes a path through the maze on the rendered maze grid.

        Args:
            path (list): A list of states (tuples of (row, column)) representing the path.
            ax (matplotlib.axes._axes.Axes, optional): An existing matplotlib Axes object to draw on.
                                                      If None, a new figure and axes are created. Defaults to None.

        Returns:
            matplotlib.axes._axes.Axes: The Axes object on which the path is visualized.
        """
        if ax is None: # If no Axes object is provided, create a new figure and Axes
            fig, ax = plt.subplots(figsize=(7, 7))

        # Render the base maze
        self.env.render(ax)

        # Plot the path as a line with markers
        path_x = [p[1] for p in path] # Extract column indices from path states
        path_y = [p[0] for p in path] # Extract row indices from path states
        ax.plot(path_x, path_y, 'o-', color='blue', markersize=10, alpha=0.6) # Plot the path with blue circles and lines

        # Number the steps along the path
        for i, (y, x) in enumerate(zip(path_y, path_x)): # Iterate through path coordinates and index
            ax.text(x, y, str(i), fontsize=12, ha='center', va='center', color='red') # Add step number text at each path point

        ax.set_title("Optimal Path") # Set plot title
        return ax # Return the Axes object

    def visualize_learning_progress(self):
        """
        Visualizes the learning progress over episodes by plotting rewards and steps per episode.

        Returns:
            matplotlib.figure.Figure: The matplotlib Figure object containing the learning progress plots.
        """
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5)) # Create a figure with two subplots side-by-side

        # Plot rewards over episodes
        window_size = min(50, len(self.rewards_history)) # Window size for rolling average, capped at 50 or episode count
        if window_size > 0: # Check if there are enough episodes to calculate rolling average
            rolling_avg = np.convolve(self.rewards_history, np.ones(window_size)/window_size, mode='valid') # Calculate rolling average of rewards
            ax1.plot(self.rewards_history, alpha=0.3, color='blue', label='Episode Reward') # Plot raw episode rewards in light blue
            ax1.plot(range(window_size-1, len(self.rewards_history)), rolling_avg, color='red', label=f'{window_size}-Episode Average') # Plot rolling average in red
            ax1.set_xlabel('Episode') # Set x-axis label
            ax1.set_ylabel('Total Reward') # Set y-axis label
            ax1.set_title('Rewards over Episodes') # Set subplot title
            ax1.legend() # Show legend

        # Plot steps per episode
        if len(self.steps_history) > 0: # Check if there is steps history to plot
            window_size = min(50, len(self.steps_history)) # Window size for rolling average of steps
            rolling_avg = np.convolve(self.steps_history, np.ones(window_size)/window_size, mode='valid') # Calculate rolling average of steps
            ax2.plot(self.steps_history, alpha=0.3, color='green', label='Steps per Episode') # Plot raw steps per episode in light green
            ax2.plot(range(window_size-1, len(self.steps_history)), rolling_avg, color='red', label=f'{window_size}-Episode Average') # Plot rolling average of steps in red
            ax2.set_xlabel('Episode') # Set x-axis label
            ax2.set_ylabel('Steps') # Set y-axis label
            ax2.set_title('Steps per Episode') # Set subplot title
            ax2.legend() # Show legend

        plt.tight_layout() # Adjust layout to prevent overlapping titles/labels
        return fig # Return the Figure object

# Main Training Loop with Visualization
env = MazeEnv() # Create an instance of the Maze environment
agent = QLearningAgent(env) # Create an instance of the Q-Learning agent, connected to the environment
episodes = 1000  # Number of episodes to train the agent for

# Set up a list to store snapshots of Q-tables for visualization at different stages of learning
q_table_snapshots = []
snapshot_intervals = [0, 10, 50, 100, 500, 999]  # Episodes at which to take snapshots of the Q-table

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
plt.savefig('learning_progress.png') # Save the learning progress plot as a PNG file
plt.close(learning_fig) # Close the figure to free up memory

# Visualize the final Q-values and the optimal path based on the learned Q-table
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 7)) # Create a figure with two subplots side-by-side
agent.visualize_q_values(ax1) # Visualize Q-values on the first subplot
optimal_path = agent.get_optimal_path() # Get the optimal path from start to goal based on learned Q-table
agent.visualize_path(optimal_path, ax2) # Visualize the optimal path on the second subplot
plt.savefig('final_result.png') # Save the final result plot as a PNG file
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
ax.set_xticks(range(5)) # Set x-ticks to column numbers
ax.set_yticks(range(5)) # Set y-ticks to row numbers

plt.savefig('q_values_3d.png') # Save the 3D Q-value heatmap as a PNG file
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
plt.savefig('learning_snapshots.png') # Save the learning snapshots plot as a PNG file
plt.show() # Display the learning snapshots plot

# Print the learned Q-table to the console for numerical inspection
print("Learned Q-table:")
print(agent.q_table)