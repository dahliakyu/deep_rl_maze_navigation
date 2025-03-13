import numpy as np
import random
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import matplotlib.patches as patches
from collections import deque
import time

class EnhancedMazeEnv:
    def __init__(self, size=12, wall_ratio=0.2):
        """
        Create a maze of given size with a specified wall ratio.
        """
        self.size = size
        self.start = (0, 0)
        self.goal = (size - 1, size - 1)
        self.maze = np.zeros((size, size), dtype=int)

        self.wall_ratio = wall_ratio
        self.num_walls = int(self.wall_ratio * size * size)

        self.current_state = self.start
        self._generate_valid_maze()

    def set_wall_ratio(self, new_ratio):
        """
        Adjust the wall ratio and regenerate a valid maze.
        """
        self.wall_ratio = new_ratio
        self.num_walls = int(self.wall_ratio * self.size * self.size)
        self._generate_valid_maze()

    def _generate_valid_maze(self, max_adjust_attempts=5):
        """
        Randomly create a maze with a valid path from start to goal.
        """
        valid_maze = False
        attempts = 0

        while not valid_maze and attempts < 200:
            self._randomize_walls()
            if self._has_valid_path():
                valid_maze = True
            else:
                for _ in range(max_adjust_attempts):
                    if self._adjust_walls() and self._has_valid_path():
                        valid_maze = True
                        break
            attempts += 1

        # Mark the goal cell explicitly
        self.goal = (self.size - 1, self.size - 1)
        self.maze[self.goal] = 9

    def _randomize_walls(self):
        """
        Fill the maze with self.num_walls randomly placed walls,
        except on start and goal.
        """
        self.maze.fill(0)
        placed = 0
        while placed < self.num_walls:
            x, y = np.random.randint(0, self.size, 2)
            if (x, y) not in [self.start, self.goal] and self.maze[x, y] == 0:
                self.maze[x, y] = 1
                placed += 1

    def _adjust_walls(self):
        """
        Removes two walls and adds two walls randomly,
        then checks if there's a valid path.
        """
        wall_positions = list(zip(*np.where(self.maze == 1)))
        if len(wall_positions) < 2:
            return False
        try:
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
        except ValueError:
            return False

    def _has_valid_path(self):
        """
        Simple DFS check to verify a path from start to goal.
        """
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

    def step(self, action, visited=None):
        """
        Reward logic:
          - Hitting a wall => -1.0
          - Valid move => -0.01
          - Reaching goal => +10.0
          - Revisiting the same cell => -0.02
        """
        wall_penalty = -1.0
        step_penalty = -0.01
        goal_reward = 10.0
        revisit_penalty = -0.02

        old_x, old_y = self.current_state
        x, y = old_x, old_y

        reward = step_penalty
        if action == 0:  # Up
            if x > 0 and self.maze[x - 1, y] != 1:
                x -= 1
            else:
                reward = wall_penalty
        elif action == 1:  # Down
            if x < self.size - 1 and self.maze[x + 1, y] != 1:
                x += 1
            else:
                reward = wall_penalty
        elif action == 2:  # Left
            if y > 0 and self.maze[x, y - 1] != 1:
                y -= 1
            else:
                reward = wall_penalty
        elif action == 3:  # Right
            if y < self.size - 1 and self.maze[x, y + 1] != 1:
                y += 1
            else:
                reward = wall_penalty

        self.current_state = (x, y)

        # Extra penalty if revisiting a cell in the same episode
        if visited is not None and self.current_state in visited:
            reward += revisit_penalty

        if self.current_state == self.goal:
            return self.current_state, goal_reward, True
        else:
            return self.current_state, reward, False

    def render(self, ax=None, show_current=False):
        if ax is None:
            fig, ax = plt.subplots(figsize=(7, 7))

        colors = ['white', 'gray', 'green']
        cmap = ListedColormap(colors)
        vis_maze = self.maze.copy()
        vis_maze[vis_maze == 9] = 2  # Mark goal as green
        ax.imshow(vis_maze, cmap=cmap)

        for i in range(self.size + 1):
            ax.axhline(i - 0.5, color='black', linewidth=1)
            ax.axvline(i - 0.5, color='black', linewidth=1)

        sx, sy = self.start
        ax.text(sy, sx, 'S', ha='center', va='center', fontsize=20, color='blue')

        gx, gy = self.goal
        ax.text(gy, gx, 'G', ha='center', va='center', fontsize=20, color='white')

        if show_current:
            cx, cy = self.current_state
            patch = patches.Circle((cy, cx), 0.3, color='red')
            ax.add_patch(patch)

        ax.set_xticks(range(self.size))
        ax.set_yticks(range(self.size))
        ax.grid(False)
        return ax


class QLearningAgent:
    def __init__(self, env, alpha=0.1, gamma=0.95, epsilon=1.0,
                 epsilon_decay=0.995, epsilon_min=0.05):
        self.env = env
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.epsilon_min = epsilon_min

        # Q-table for a 12x12 environment
        self.q_table = np.zeros((env.size, env.size, 4))

        # Trackers
        self.rewards_history = []
        self.steps_history = []
        self.times_history = []
        self.success_history = []

    def choose_action(self, state):
        if random.random() < self.epsilon:
            return random.randint(0, 3)
        x, y = state
        return np.argmax(self.q_table[x, y])

    def update_q_value(self, state, action, reward, next_state):
        x, y = state
        nx, ny = next_state
        max_q_next = np.max(self.q_table[nx, ny])
        self.q_table[x, y, action] += self.alpha * (
            reward + self.gamma * max_q_next - self.q_table[x, y, action]
        )

    def get_optimal_path(self, max_steps=200):
        """
        Trace a path using greedy actions from the Q-table.
        """
        path = []
        state = self.env.reset()
        done = False
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

    def visualize_q_values(self, ax=None, title="Q-values and Optimal Actions"):
        """
        Displays the maze with arrows indicating best actions and Q-values.
        """
        if ax is None:
            fig, ax = plt.subplots(figsize=(7, 7))
        self.env.render(ax)

        rows, cols, _ = self.q_table.shape
        for i in range(rows):
            for j in range(cols):
                # Skip walls
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
                ax.arrow(j, i, dx, dy, head_width=0.2, head_length=0.2,
                         fc=arrow_color, ec=arrow_color, width=0.05)

                # Q-value text offset
                tx = j + (0.2 if dx < 0 else (-0.2 if dx > 0 else 0))
                ty = i + (0.2 if dy < 0 else (-0.2 if dy > 0 else 0))
                ax.text(tx, ty, f"{q_value:.1f}",
                        fontsize=8, ha='center', va='center',
                        bbox=dict(facecolor='white', alpha=0.5, pad=1))

        ax.set_title(title)
        return ax

    def visualize_path(self, path, ax=None):
        """
        Draws the agent's path on top of the current maze layout.
        """
        if ax is None:
            fig, ax = plt.subplots(figsize=(7, 7))
        self.env.render(ax)
        px = [p[1] for p in path]
        py = [p[0] for p in path]
        ax.plot(px, py, 'o-', color='blue', markersize=8, alpha=0.7)
        for i, (yy, xx) in enumerate(zip(py, px)):
            ax.text(xx, yy, str(i), fontsize=10, ha='center', va='center', color='red')
        ax.set_title("Agent Path")
        return ax


# --------------------------------------
# Utility to capture Q-value snapshot
# --------------------------------------
def capture_q_snapshot(agent, title=""):
    """
    Renders the Q-values in a small figure and returns an image array.
    """
    fig, ax = plt.subplots(figsize=(4, 4))
    agent.visualize_q_values(ax, title=title)
    fig.canvas.draw()
    img_array = np.array(fig.canvas.renderer._renderer)
    if img_array.shape[2] == 4:
        img_array = img_array[:, :, :3]  # drop alpha channel if present
    plt.close(fig)
    return img_array


def create_collage(snapshots, ratios_order):
    """
    Creates a collage from the snapshots dictionary,
    where snapshots[ratio] = (episode, image_array).
    We'll display them in the order given by ratios_order.
    """
    n = len(ratios_order)
    fig, axes = plt.subplots(nrows=1, ncols=n, figsize=(4*n, 4))

    if n == 1:
        axes = [axes]  # make it iterable

    for i, ratio in enumerate(ratios_order):
        ax = axes[i]
        ep, img_array = snapshots[ratio]
        ax.imshow(img_array)
        # For display, multiply ratio by 100
        ax.set_title(f"{int(ratio*100)}% Walls (Ep={ep})")
        ax.axis('off')

    fig.tight_layout()
    return fig


if __name__ == "__main__":

    wall_ratios = [0.2, 0.3, 0.4, 0.5]
    ratio_index = 0

    env = EnhancedMazeEnv(size=12, wall_ratio=wall_ratios[ratio_index])
    agent = QLearningAgent(env)

    max_episodes = 5000
    max_steps_per_episode = 300

    # Rolling success window is now 50 episodes
    success_deque = deque(maxlen=50)
    success_threshold = 0.8

    snapshots = {}

    for episode in range(1, max_episodes + 1):
        start_time = time.time()
        state = env.reset()
        visited_states = set([state])
        done = False
        total_reward = 0
        steps = 0

        while not done:
            action = agent.choose_action(state)
            next_state, reward, done = env.step(action, visited=visited_states)
            agent.update_q_value(state, action, reward, next_state)
            state = next_state
            visited_states.add(state)
            total_reward += reward
            steps += 1

            if steps >= max_steps_per_episode and not done:
                total_reward -= 5.0
                done = True

        agent.epsilon = max(agent.epsilon_min, agent.epsilon * agent.epsilon_decay)

        episode_time = time.time() - start_time
        agent.rewards_history.append(total_reward)
        agent.steps_history.append(steps)
        agent.times_history.append(episode_time)
        agent.success_history.append(1 if (state == env.goal) else 0)
        success_deque.append(1 if (state == env.goal) else 0)

        # Now we check if we have 50 episodes in the success window
        if len(success_deque) == 50:
            success_rate = sum(success_deque) / 50.0
            # If threshold is reached, move to next ratio
            if success_rate >= success_threshold and ratio_index < len(wall_ratios) - 1:
                current_ratio = wall_ratios[ratio_index]
                snap_img = capture_q_snapshot(agent, title=f"{int(current_ratio*100)}% Walls (before ratio change)")
                snapshots[current_ratio] = (episode, snap_img)

                new_ratio = wall_ratios[ratio_index + 1]
                print(
                    f"Success threshold of {success_rate*100:.1f}% reached at Episode {episode}.\n"
                    f"Raising walls from {int(current_ratio*100)}% to {int(new_ratio*100)}%."
                )

                ratio_index += 1
                env.set_wall_ratio(new_ratio)
                success_deque.clear()

        # Print logs every 100 episodes (unchanged from before)
        if episode % 100 == 0:
            recent_rewards = agent.rewards_history[-100:]
            avg_reward = sum(recent_rewards) / len(recent_rewards)
            recent_steps = agent.steps_history[-100:]
            avg_steps = sum(recent_steps) / len(recent_steps)
            recent_success = agent.success_history[-100:]
            success_rate_100 = sum(recent_success) / len(recent_success) * 100
            print(f"[Episode {episode}] Ratio: {env.wall_ratio*100:.0f}%, "
                  f"AvgReward: {avg_reward:.2f}, AvgSteps: {avg_steps:.2f}, "
                  f"SuccessRate(100ep): {success_rate_100:.1f}%")

        # If at final ratio, stop if threshold is again met
        if ratio_index == len(wall_ratios) - 1:
            if len(success_deque) == 50:
                final_success_rate = sum(success_deque) / 50.0
                if final_success_rate >= success_threshold:
                    print(f"Reached {success_threshold*100:.0f}% success at {env.wall_ratio*100:.0f}% walls. Stopping at episode={episode}")
                    current_ratio = wall_ratios[ratio_index]
                    snap_img = capture_q_snapshot(agent, title=f"{int(current_ratio*100)}% Walls (before stopping)")
                    snapshots[current_ratio] = (episode, snap_img)
                    break

    # If training ends without hitting final break,
    # take a snapshot of the final ratio
    if wall_ratios[ratio_index] not in snapshots:
        snap_img = capture_q_snapshot(agent, title=f"{int(wall_ratios[ratio_index]*100)}% Walls (end)")
        snapshots[wall_ratios[ratio_index]] = (max_episodes, snap_img)

    used_ratios = sorted(snapshots.keys())
    collage_fig = create_collage(snapshots, used_ratios)
    plt.show()
    collage_fig.savefig("./curriculum_results/ratio_milestones_collage.png")
    plt.close(collage_fig)

    # Save & show progress figure
    fig_progress, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(20, 5))
    ax1.plot(agent.rewards_history, alpha=0.5)
    ax1.set_xlabel('Episode')
    ax1.set_ylabel('Reward')
    ax1.set_title("Reward per Episode")

    ax2.plot(agent.steps_history, alpha=0.5)
    ax2.set_xlabel('Episode')
    ax2.set_ylabel('Steps')
    ax2.set_title("Steps per Episode")

    window_size = 50
    success_moving_avg = []
    for i in range(len(agent.success_history)):
        window = agent.success_history[max(0, i - window_size + 1):i + 1]
        success_moving_avg.append(sum(window) / len(window))
    ax3.plot(success_moving_avg)
    ax3.set_xlabel('Episode')
    ax3.set_ylabel('Success Rate')
    ax3.set_title(f"Success Rate (Moving Avg over {window_size} eps)")

    fig_progress.tight_layout()
    plt.show()
    fig_progress.savefig("./curriculum_results/progress.png")
    plt.close(fig_progress)

    # Save & show final Q-values and path
    fig_final, (axA, axB) = plt.subplots(1, 2, figsize=(14, 7))
    agent.visualize_q_values(axA, title="Final Q-Values")
    path = agent.get_optimal_path()
    agent.visualize_path(path, axB)
    plt.show()
    fig_final.savefig("./curriculum_results/final_q_and_path.png")
    plt.close(fig_final)
