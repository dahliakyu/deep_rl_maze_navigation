import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np

class Visualization:
    """
    Handles advanced visualizations of the maze environment and agent's performance.
    Delegates basic visualizations to the agent.
    """

    def __init__(self, env, agent):
        """
        Initializes the visualization.

        Args:
            env (MazeEnv): The maze environment to visualize.
            agent (Agent): The agent being visualized.
        """
        self.env = env
        self.agent = agent

    def visualize_learning(self, rewards_per_episode):
        """
        Visualizes the learning progress (rewards over episodes) and saves the plot.
        """
        learning_fig = self.agent.visualize_learning_progress(rewards_per_episode)
        plt.savefig('learning_progress.png')
        plt.close(learning_fig)

    def visualize_final_result(self):
        """
        Visualizes the final Q-values and the optimal path and saves the plot.
        """
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 7))
        im = self.agent.visualize_q_values(ax1)
        fig.colorbar(im, ax=ax1, shrink=0.7)
        optimal_path = self.agent.get_optimal_path()
        self.agent.visualize_path(optimal_path, ax2)
        plt.savefig('final_result.png')
        plt.close(fig)

    def visualize_q_values_3d(self):
        """
        Creates a 3D heatmap visualization of the Q-values and saves the plot.
        """
        fig = plt.figure(figsize=(15, 10))
        ax = fig.add_subplot(111, projection='3d')

        max_q_values = np.zeros((self.env.grid_size[0], self.env.grid_size[1]))
        for i in range(self.env.grid_size[0]):
            for j in range(self.env.grid_size[1]):
                state = (i, j)
                if state in self.agent.q_table:
                    max_q_values[i, j] = np.max(self.agent.q_table[state])
                else:
                    max_q_values[i, j] = 0

        x = np.arange(self.env.grid_size[1])  # Corrected to grid_size[1]
        y = np.arange(self.env.grid_size[0])  # Corrected to grid_size[0]
        x, y = np.meshgrid(x, y)
        z = max_q_values.copy()

        for i in range(self.env.grid_size[0]):
            for j in range(self.env.grid_size[1]):
                if self.env.maze[i, j] == 1:
                    z[i, j] = 0

        surf = ax.plot_surface(x, y, z, cmap='viridis', edgecolor='none', alpha=0.8)

        fig.colorbar(surf, ax=ax, shrink=0.5, aspect=5)
        ax.set_xlabel('Y Coordinate')  # Corrected label
        ax.set_ylabel('X Coordinate')  # Corrected label
        ax.set_zlabel('Max Q-Value')
        ax.set_title('3D Heatmap of Q-Values')

        ax.set_xticks(range(self.env.grid_size[1]))  # Corrected ticks
        ax.set_yticks(range(self.env.grid_size[0]))  # Corrected ticks

        plt.savefig('q_values_3d.png')
        plt.close(fig)

    def visualize_learning_snapshots(self, q_table_snapshots):
        """
        Visualizes snapshots of learning at different episodes and saves the plot.
        """
        rows = 2
        cols = 3
        fig, axes = plt.subplots(rows, cols, figsize=(18, 10))
        axes = axes.flatten()

        for i, (episode, q_table) in enumerate(q_table_snapshots):
            original_q_table = self.agent.q_table.copy()  # Store the original Q-table
            self.agent.q_table = q_table  # Temporarily set the agent's Q-table to the snapshot Q-table

            #im = axes[i].imshow(self.agent.visualize_q_values(axes[i]))
            im = self.agent.visualize_q_values(axes[i])
            fig.colorbar(im, ax=axes[i], shrink=0.7)
            axes[i].set_title(f"Episode {episode}")
            axes[i].invert_yaxis()

            self.agent.q_table = original_q_table  # Restore the original Q-table

        plt.tight_layout()
        plt.savefig('learning_snapshots.png')
        plt.close(fig)