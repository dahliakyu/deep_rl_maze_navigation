import numpy as np
import random
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import matplotlib.patches as patches


class EnhancedMazeEnv:
    def __init__(self, size=5, num_walls=7):
        self.size = size
        self.num_walls = num_walls
        self.start = (0, 0)
        self.goal = (size - 1, size - 1)
        self.maze = np.zeros((size, size))
        self.current_state = self.start
        self.generate_valid_maze()

    def generate_valid_maze(self, max_adjust_attempts=3):
        valid_maze = False

        while not valid_maze:
            self._randomize_walls()
            if self._has_valid_path():
                valid_maze = True
                break

            for _ in range(max_adjust_attempts):
                if self._adjust_walls() and self._has_valid_path():
                    valid_maze = True
                    break

        self.maze[self.goal] = 9

    def _randomize_walls(self):
        self.maze = np.zeros((self.size, self.size))
        placed = 0
        while placed < self.num_walls:
            x, y = np.random.randint(0, self.size, 2)
            if (x, y) not in [self.start, self.goal] and self.maze[x, y] != 1:
                self.maze[x, y] = 1
                placed += 1

    def _adjust_walls(self):
        try:
            wall_positions = list(zip(*np.where(self.maze == 1)))
            removed = random.sample(wall_positions, 2)
            for x, y in removed:
                self.maze[x, y] = 0

            added = 0
            while added < 2:
                x, y = np.random.randint(0, self.size, 2)
                if (x, y) not in [self.start, self.goal] and self.maze[x, y] == 0:
                    self.maze[x, y] = 1
                    added += 1
            return True
        except:
            return False

    def _has_valid_path(self):
        visited = set()
        stack = [self.start]

        while stack:
            x, y = stack.pop()

            if (x, y) == self.goal:
                return True

            if (x, y) in visited:
                continue

            visited.add((x, y))

            for dx, dy in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                nx, ny = x + dx, y + dy
                if 0 <= nx < self.size and 0 <= ny < self.size:
                    if self.maze[nx, ny] != 1 and (nx, ny) not in visited:
                        stack.append((nx, ny))

        return False

    def reset(self):
        self.current_state = self.start
        return self.current_state

    def step(self, action):
        x, y = self.current_state
        if action == 0:  # Up
            if x > 0 and self.maze[x - 1, y] != 1:
                x -= 1
        elif action == 1:  # Down
            if x < self.size - 1 and self.maze[x + 1, y] != 1:
                x += 1
        elif action == 2:  # Left
            if y > 0 and self.maze[x, y - 1] != 1:
                y -= 1
        elif action == 3:  # Right
            if y < self.size - 1 and self.maze[x, y + 1] != 1:
                y += 1

        self.current_state = (x, y)

        if self.maze[x, y] == 9:
            return self.current_state, 10, True
        elif self.maze[x, y] == 1:
            return self.current_state, -1, False
        else:
            return self.current_state, -0.1, False

    def render(self, ax=None, show_current=False):
        if ax is None:
            fig, ax = plt.subplots(figsize=(7, 7))

        colors = ['white', 'gray', 'green']
        cmap = ListedColormap(colors)
        vis_maze = self.maze.copy()
        vis_maze[vis_maze == 9] = 2

        ax.imshow(vis_maze, cmap=cmap)

        for i in range(self.size + 1):
            ax.axhline(i - 0.5, color='black', linewidth=1)
            ax.axvline(i - 0.5, color='black', linewidth=1)

        start_y, start_x = self.start
        ax.text(start_x, start_y, 'S', ha='center', va='center', fontsize=20, color='blue')

        goal_idx = np.where(self.maze == 9)
        if len(goal_idx[0]) > 0:
            goal_y, goal_x = goal_idx[0][0], goal_idx[1][0]
            ax.text(goal_x, goal_y, 'G', ha='center', va='center', fontsize=20, color='white')

        if show_current:
            curr_y, curr_x = self.current_state
            ax.add_patch(patches.Circle((curr_x, curr_y), 0.3, color='red'))

        ax.set_xticks(range(self.size))
        ax.set_yticks(range(self.size))
        ax.grid(False)
        return ax


class QLearningAgent:
    def __init__(self, env, alpha=0.1, gamma=0.9, epsilon=0.1):
        self.env = env
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon
        self.q_table = np.zeros((env.size, env.size, 4))
        self.rewards_history = []
        self.steps_history = []

    def choose_action(self, state):
        if random.uniform(0, 1) < self.epsilon:
            return random.randint(0, 3)
        else:
            x, y = state
            return np.argmax(self.q_table[x, y])

    def update_q_value(self, state, action, reward, next_state):
        x, y = state
        next_x, next_y = next_state
        max_q_next = np.max(self.q_table[next_x, next_y])
        self.q_table[x, y, action] += self.alpha * (reward + self.gamma * max_q_next - self.q_table[x, y, action])

    def get_optimal_path(self):
        path = []
        state = self.env.reset()
        done = False
        max_steps = 25
        step_count = 0

        while not done and step_count < max_steps:
            path.append(state)
            x, y = state
            action = np.argmax(self.q_table[x, y])
            next_state, _, done = self.env.step(action)
            state = next_state
            step_count += 1

        path.append(state)
        return path

    def visualize_q_values(self, ax=None):
        if ax is None:
            fig, ax = plt.subplots(figsize=(10, 10))

        self.env.render(ax)

        for i in range(self.env.size):
            for j in range(self.env.size):
                if self.env.maze[i, j] == 1:
                    continue

                best_action = np.argmax(self.q_table[i, j])
                q_value = np.max(self.q_table[i, j])

                if q_value <= 0:
                    continue

                dx, dy = 0, 0
                if best_action == 0:
                    dx, dy = 0, -0.4
                elif best_action == 1:
                    dx, dy = 0, 0.4
                elif best_action == 2:
                    dx, dy = -0.4, 0
                elif best_action == 3:
                    dx, dy = 0.4, 0

                arrow_color = plt.cm.viridis(min(q_value / 10, 1))
                ax.arrow(j, i, dx, dy, head_width=0.2, head_length=0.2, fc=arrow_color, ec=arrow_color, width=0.05)

                if q_value > 0:
                    text_offset_x = -0.2 if dx > 0 else 0.2 if dx < 0 else 0
                    text_offset_y = -0.2 if dy > 0 else 0.2 if dy < 0 else 0
                    ax.text(j + text_offset_x, i + text_offset_y, f"{q_value:.1f}",
                            fontsize=8, ha='center', va='center',
                            bbox=dict(facecolor='white', alpha=0.5, pad=1))

        ax.set_title("Q-values and Optimal Actions")
        return ax

    def visualize_learning_progress(self):
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))

        window_size = min(50, len(self.rewards_history))
        if window_size > 0:
            rolling_avg = np.convolve(self.rewards_history, np.ones(window_size) / window_size, mode='valid')
            ax1.plot(self.rewards_history, alpha=0.3, color='blue', label='Episode Reward')
            ax1.plot(range(window_size - 1, len(self.rewards_history)), rolling_avg, color='red',
                     label=f'{window_size}-Episode Average')
            ax1.set_xlabel('Episode')
            ax1.set_ylabel('Total Reward')
            ax1.legend()

        if len(self.steps_history) > 0:
            window_size = min(50, len(self.steps_history))
            rolling_avg = np.convolve(self.steps_history, np.ones(window_size) / window_size, mode='valid')
            ax2.plot(self.steps_history, alpha=0.3, color='green', label='Steps per Episode')
            ax2.plot(range(window_size - 1, len(self.steps_history)), rolling_avg, color='red',
                     label=f'{window_size}-Episode Average')
            ax2.set_xlabel('Episode')
            ax2.set_ylabel('Steps')
            ax2.legend()

        plt.tight_layout()
        return fig

    def visualize_path(self, path, ax=None):
        """Visualize a path through the maze"""
        if ax is None:
            fig, ax = plt.subplots(figsize=(7, 7))

        # Render the base maze
        self.env.render(ax)

        # Plot the path
        path_x = [p[1] for p in path]  # Column index (x-coordinate)
        path_y = [p[0] for p in path]  # Row index (y-coordinate)
        ax.plot(path_x, path_y, 'o-', color='blue', markersize=10, alpha=0.6)

        # Number the steps
        for i, (y, x) in enumerate(zip(path_y, path_x)):
            ax.text(x, y, str(i), fontsize=12, ha='center', va='center', color='red')

        ax.set_title("Optimal Path")
        return ax


if __name__ == "__main__":
    # Initialize environment and agent
    env = EnhancedMazeEnv(size=5, num_walls=7)
    agent = QLearningAgent(env)
    episodes = 1000
    q_table_snapshots = []
    snapshot_intervals = [0, 10, 50, 100, 500, 999]  # For learning snapshots

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

        agent.rewards_history.append(total_reward)
        agent.steps_history.append(steps)

        # Store Q-table snapshots
        if episode in snapshot_intervals:
            q_table_snapshots.append((episode, agent.q_table.copy()))

        if episode % 100 == 0:
            print(f"Episode {episode}: Reward={total_reward}, Steps={steps}")

    # Save learning progress plot
    learning_fig = agent.visualize_learning_progress()
    plt.savefig('simple_q_dfs_maze_results/learning_progress.png')
    plt.close(learning_fig)

    # Save final Q-values and path
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 7))
    agent.visualize_q_values(ax1)
    agent.visualize_path(agent.get_optimal_path(), ax2)
    plt.savefig('simple_q_dfs_maze_results/final_result.png')
    plt.close(fig)

    # Save 3D heatmap
    fig = plt.figure(figsize=(15, 10))
    ax = fig.add_subplot(111, projection='3d')
    max_q_values = np.max(agent.q_table, axis=2)
    x, y = np.meshgrid(np.arange(5), np.arange(5))
    z = max_q_values.copy()

    # Remove walls from visualization
    for i in range(5):
        for j in range(5):
            if env.maze[i, j] == 1:
                z[i, j] = 0

    surf = ax.plot_surface(x, y, z, cmap='viridis', edgecolor='none', alpha=0.8)
    fig.colorbar(surf, ax=ax, shrink=0.5, aspect=5)
    ax.set(xlabel='Y', ylabel='X', zlabel='Max Q-Value', title='3D Q-Value Heatmap')
    plt.savefig('simple_q_dfs_maze_results/q_values_3d.png')
    plt.close(fig)

    # Save learning snapshots
    rows, cols = 2, 3
    fig, axes = plt.subplots(rows, cols, figsize=(18, 10))
    axes = axes.flatten()

    for i, (episode, q_table) in enumerate(q_table_snapshots):
        original_q = agent.q_table.copy()
        agent.q_table = q_table
        agent.visualize_q_values(axes[i])
        axes[i].set_title(f"Episode {episode}")
        agent.q_table = original_q

    plt.tight_layout()
    plt.savefig('simple_q_dfs_maze_results/learning_snapshots.png')
    plt.close()

    print("All results saved to simple_q_dfs_maze_results directory")