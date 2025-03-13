from abc import ABC, abstractmethod
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap

class Agent(ABC):
    """
    Abstract base class for all agents.
    Defines the basic interface for interacting with the environment.
    """

    def __init__(self, env):
        """
        Initializes the agent.

        Args:
            env (MazeEnv): The environment the agent will interact with.
        """
        self.env = env
        self.q_table = None  # Initialize q_table to None

    @abstractmethod
    def choose_action(self, state):
        """
        Chooses an action based on the current state.
        This method must be implemented by subclasses.

        Args:
            state (tuple): The current state of the environment.

        Returns:
            int: The action to take.
        """
        pass

    @abstractmethod
    def learn(self, state, action, reward, next_state, done):
        """
        Updates the agent's knowledge based on the experience.
        This method must be implemented by subclasses.

        Args:
            state (tuple): The state the agent was in.
            action (int): The action the agent took.
            reward (float): The reward received after taking the action.
            next_state (tuple): The next state the agent transitioned to.
            done (bool): Whether the episode is finished.
        """
        pass

    def visualize_learning_progress(self, rewards_per_episode):
        """
        Visualizes the learning progress by plotting rewards over episodes.

        Args:
            rewards_per_episode (list): A list of rewards obtained in each episode.

        Returns:
            matplotlib.figure.Figure: The matplotlib figure containing the plot.
        """
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.plot(rewards_per_episode)
        ax.set_xlabel("Episode")
        ax.set_ylabel("Total Reward")
        ax.set_title("Learning Progress")
        return fig

    def visualize_q_values(self, ax):
        """
        Visualizes only the highest Q-value (if it's > 1) with a larger arrow.

        Args:
            ax (matplotlib.axes._axes.Axes): The axes object to draw on.
        """
        grid_size = self.env.grid_size
        actions = ["Up", "Down", "Left", "Right"]  # Action labels
        ax.clear() # Clear axes

        # Render the base environment (maze)
        cmap = ListedColormap(['white', 'gray', 'green'])
        vis_maze = self.env.maze.copy()
        vis_maze[vis_maze == 9] = 2
        ax.imshow(vis_maze, cmap=cmap, origin='upper')

        # Iterate through each state
        for i in range(grid_size[0]):
            for j in range(grid_size[1]):
                state = (i, j)
                if state in self.q_table:
                    q_values = self.q_table[state]
                    best_action_idx = np.argmax(q_values)
                    best_q_value = q_values[best_action_idx]

                    # Only display if the best Q-value is greater than 1
                    if best_q_value > 1:
                        dx, dy = 0, 0
                        arrow_scale = 0.4  # Increased scale for better visibility
                        if best_action_idx == 0:  # Up
                            dy = -arrow_scale
                        elif best_action_idx == 1:  # Down
                            dy = arrow_scale
                        elif best_action_idx == 2:  # Left
                            dx = -arrow_scale
                        elif best_action_idx == 3:  # Right
                            dx = arrow_scale

                        arrow_color = 'blue' # Change the arrow color

                        # Draw the arrow for the best action
                        ax.arrow(j, i, dx, dy, head_width=0.15, head_length=0.15, fc=arrow_color, ec=arrow_color, alpha=0.8)  # Adjust head dimensions

                        # Display the Q-value next to the arrow
                        ax.text(j + 1.2 * dx, i + 1.2 * dy, f"{best_q_value:.1f}", color='black', ha='center', va='center', fontsize=10)

        ax.set_xticks(np.arange(grid_size[1]))
        ax.set_yticks(np.arange(grid_size[0]))
        ax.set_xticklabels(np.arange(grid_size[1]))
        ax.set_yticklabels(np.arange(grid_size[0]))
        ax.set_title("Best Q-Values (> 1) and Policy")
        ax.invert_yaxis() # Invert

    def get_optimal_path(self):
        """
        Calculates the optimal path from start to goal based on the learned Q-table.

        Returns:
            list: A list of states representing the optimal path.
        """
        start_state = self.env.start
        goal_state = self.env.goal
        current_state = start_state
        path = [current_state]

        while current_state != goal_state:
            if current_state not in self.q_table:
                return None  # No path found

            best_action = np.argmax(self.q_table[current_state])

            # Determine the next state based on the action (you'll need to adapt this)
            x, y = current_state
            if best_action == 0:  # Up
                next_state = (max(0, x - 1), y)
            elif best_action == 1:  # Down
                next_state = (min(self.env.grid_size[0] - 1, x + 1), y)
            elif best_action == 2:  # Left
                next_state = (x, max(0, y - 1))
            elif best_action == 3:  # Right
                next_state = (x, min(self.env.grid_size[1] - 1, y + 1))
            else:
                return None # No path found

            path.append(next_state)
            current_state = next_state
            if len(path) > 100:
                 return None

        return path

    def visualize_path(self, path, ax):
        """
        Visualizes the given path on the maze environment.

        Args:
            path (list): A list of states representing the path.
            ax (matplotlib.axes._axes.Axes): The axes object to draw on.
        """
        grid_size = self.env.grid_size
        cmap = ListedColormap(['white', 'gray', 'green'])
        vis_maze = self.env.maze.copy()
        vis_maze[vis_maze == 9] = 2

        ax.imshow(vis_maze, cmap=cmap, origin='upper')

        # Mark the path
        if path:
            path_x, path_y = zip(*path)
            ax.plot(path_y, path_x, marker='o', color='red', markersize=8, linestyle='-', linewidth=2)

        # Mark the start and goal
        ax.text(self.env.start[1], self.env.start[0], 'S', ha='center', va='center', color='blue', fontsize=12)
        ax.text(self.env.goal[1], self.env.goal[0], 'G', ha='center', va='center', color='white', fontsize=12)

        ax.set_title("Optimal Path")
        ax.set_xticks(np.arange(grid_size[1]))
        ax.set_yticks(np.arange(grid_size[0]))
        ax.set_xticklabels(np.arange(grid_size[1]))
        ax.set_yticklabels(np.arange(grid_size[0]))