import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import matplotlib.patches as patches
import json

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
        """Unchanged from original logic, but with proximity reward"""
        x, y = self.current_state
        new_x, new_y = x, y

        if action == 0:  # Up
            new_x = max(0, x-1)
        elif action == 1:  # Down
            new_x = min(self.size-1, x+1)
        elif action == 2:  # Left
            new_y = max(0, y-1)
        elif action == 3:  # Right
            new_y = min(self.size-1, y+1)

        # Check for walls
        if self.maze[new_x, new_y] != 1:
            self.current_state = (new_x, new_y)

        # Calculate reward

        reward = -0.01 # Small penalty for each step

        # Calculate distance to goal before and after the step
        distance_before = np.linalg.norm(np.array( (x, y)) - np.array(self.goal))
        distance_after  = np.linalg.norm(np.array(self.current_state) - np.array(self.goal))

        proximity_reward = 0.0  # Initialize proximity reward
        if distance_after < distance_before: # Getting closer to goal
            proximity_reward = 0.1 * (distance_before - distance_after)  # Small reward for proximity

        reward += proximity_reward


        if self.current_state == self.goal:
            reward = 100 # Huge reward for reaching goal
            return self.current_state, reward, True
        elif self.maze[new_x, new_y] == 1:
            reward = -5  # Large penalty for hitting a wall
            return self.current_state, reward, False
        else:
            return self.current_state, reward, False
        
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
            ax.text(goal_x, goal_y, 'G', ha='center', va='center', fontsize=20, color='red') # Add 'G' text at goal position

        # Show current position if requested
        if show_current: # If show_current is True
            curr_y, curr_x = self.current_state # Get current row and column
            ax.add_patch(patches.Circle((curr_x, curr_y), 0.3, color='red')) # Add a red circle to represent current position

        # Label the axes with coordinates (optional, for clarity)
        ax.set_xticks(range(self.maze.shape[1])) # Set x-axis ticks to column numbers
        ax.set_yticks(range(self.maze.shape[0])) # Set y-axis ticks to row numbers
        ax.grid(False) # Turn off default grid lines, as we've drawn our own

        return ax # Return the Axes object for further modifications if needed

class ComplexMazeEnv:
    def __init__(self, maze_config):
        """
        Modified to load maze from JSON file
        """
        if isinstance(maze_config, np.ndarray):
            self.maze = maze_config
        elif isinstance(maze_config, str):
            self.maze = np.array(json.load(open(maze_config, 'r')))        
        self.size = self.maze.shape[0]
        self.start = (0, 0)
        self.goal = (self.size-1, self.size-1)
        self.current_state = self.start

    def reset(self):
        """Unchanged"""
        self.current_state = self.start
        return self.current_state

    def step(self, action):
        """Unchanged from original logic, but with proximity reward"""
        x, y = self.current_state
        new_x, new_y = x, y

        if action == 0:  # Up
            new_x = max(0, x-1)
        elif action == 1:  # Down
            new_x = min(self.size-1, x+1)
        elif action == 2:  # Left
            new_y = max(0, y-1)
        elif action == 3:  # Right
            new_y = min(self.size-1, y+1)

        # Check for walls
        if self.maze[new_x, new_y] != 1:
            self.current_state = (new_x, new_y)

        # Calculate reward

        reward = -0.01 # Small penalty for each step

        # # Calculate distance to goal before and after the step
        # distance_before = np.linalg.norm(np.array( (x, y)) - np.array(self.goal))
        # distance_after  = np.linalg.norm(np.array(self.current_state) - np.array(self.goal))

        # proximity_reward = 0.0  # Initialize proximity reward
        # if distance_after < distance_before: # Getting closer to goal
        #     proximity_reward = 0.1 * (distance_before - distance_after)  # Small reward for proximity

        # reward += proximity_reward


        if self.current_state == self.goal:
            reward = 100 # Huge reward for reaching goal
            return self.current_state, reward, True
        elif self.maze[new_x, new_y] == 1:
            reward = -5  # Large penalty for hitting a wall
            return self.current_state, reward, False
        else:
            return self.current_state, reward, False

    def render(self, ax=None, show_current=False):
        """Unchanged from original visualization"""
        if ax is None:
            fig, ax = plt.subplots(figsize=(10, 10))

        colors = ['white', 'gray', 'green']
        cmap = ListedColormap(colors)
        
        ax.imshow(self.maze, cmap=cmap, vmin=0, vmax=2)
        
        # Add grid lines
        for i in range(self.size+1):
            ax.axhline(i-0.5, color='black', linewidth=1)
            ax.axvline(i-0.5, color='black', linewidth=1)
            
        # Mark start and goal
        ax.text(0, 0, 'S', ha='center', va='center', 
                fontsize=20, color='blue')
        ax.text(self.size-1, self.size-1, 'G', ha='center', va='center', 
                fontsize=20, color='red')

        # Show current position
        if show_current:
            curr_y, curr_x = self.current_state
            ax.add_patch(patches.Circle((curr_x, curr_y), 0.3, color='red'))

        ax.set_xticks(range(self.size))
        ax.set_yticks(range(self.size))
        ax.grid(False)
        return ax
