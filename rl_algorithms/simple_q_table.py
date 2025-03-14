import matplotlib.pyplot as plt
import random
import numpy as np

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
        self.q_table = np.zeros((9, 9, 4))  # Q-table initialized to zeros. Dimensions: (rows, columns, actions)
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
        max_steps = 50  # Maximum steps to prevent infinite loops if no path is found or agent gets stuck
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
        for i in range(9): # Iterate through rows
            for j in range(9): # Iterate through columns
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
