import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import matplotlib.patches as patches

class MazeEnv:
    def __init__(self, grid_size=(5, 5), walls=None, start=(0, 0), goal=(4, 4)):
        """
        Initializes the Maze environment.

        Args:
            grid_size (tuple): The size of the maze grid (rows, columns).
            walls (list of tuples): A list of (row, column) tuples representing wall locations.
            start (tuple): The starting position (row, column).
            goal (tuple): The goal position (row, column).
        """
        self.grid_size = grid_size
        self.maze = np.zeros(grid_size)
        self.walls = walls if walls else []  # Default to empty list if no walls provided
        self.start = start
        self.goal = goal

        # Initialize maze
        for wall in self.walls:
            self.maze[wall] = 1
        self.maze[self.goal] = 9  # Mark the goal

        self.current_state = self.start
        self.action_space = [0, 1, 2, 3]  # Up, Down, Left, Right

    def reset(self):
        """Resets the environment to the starting state."""
        self.current_state = self.start
        return self.current_state

    def step(self, action):
        """
        Executes an action in the environment.

        Args:
            action (int): The action to take (0: Up, 1: Down, 2: Left, 3: Right).

        Returns:
            tuple: (next_state, reward, done)
        """
        x, y = self.current_state
        new_x, new_y = x, y

        if action == 0:  # Up
            new_x = max(0, x - 1)
        elif action == 1:  # Down
            new_x = min(self.grid_size[0] - 1, x + 1)
        elif action == 2:  # Left
            new_y = max(0, y - 1)
        elif action == 3:  # Right
            new_y = min(self.grid_size[1] - 1, y + 1)

        #Check wall
        if (new_x,new_y) not in self.walls:
             self.current_state = (new_x, new_y)
        else:
            new_x, new_y = x, y
            self.current_state = (new_x, new_y)


        reward = -0.1
        done = False

        if self.current_state == self.goal:
            reward = 10
            done = True
        elif (new_x, new_y) in self.walls:
            reward = -1


        return self.current_state, reward, done

    def render(self, ax=None, show_current=False):
        """
        Renders the maze environment visually.

        Args:
            ax (matplotlib.axes._axes.Axes, optional): An existing matplotlib Axes object.
            show_current (bool, optional): Show the current agent position.
        """
        if ax is None:
            fig, ax = plt.subplots(figsize=(7, 7))

        colors = ['white', 'gray', 'green']
        cmap = ListedColormap(colors)

        vis_maze = self.maze.copy()
        vis_maze[vis_maze == 9] = 2

        ax.imshow(vis_maze, cmap=cmap, origin='upper')  # Ensure correct orientation

        # Plot grid lines
        for i in range(self.grid_size[0] + 1):
            ax.axhline(i - 0.5, color='black', linewidth=1)
            ax.axvline(i - 0.5, color='black', linewidth=1)

        # Mark start and goal
        ax.text(self.start[1], self.start[0], 'S', ha='center', va='center', fontsize=20, color='blue')  # Corrected order
        ax.text(self.goal[1], self.goal[0], 'G', ha='center', va='center', fontsize=20, color='white')  # Corrected order

        # Show current position
        if show_current:
            curr_x, curr_y = self.current_state[0], self.current_state[1]
            ax.add_patch(patches.Circle((curr_y, curr_x), curr_x, color='red'))  # Corrected order


        ax.set_xticks(range(self.grid_size[1]))
        ax.set_yticks(range(self.grid_size[0]))
        ax.invert_yaxis() #invert the y axis


        return ax