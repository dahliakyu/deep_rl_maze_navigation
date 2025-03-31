import json
import numpy as np
import matplotlib.pyplot as plt

def analyze_and_visualize_maze_data(file_paths, save_plots=True):
    """
    Loads data from multiple JSON files, plots individual reward and step histories
    as dot plots, and the overall average with standard deviation as a line plot
    with error shading, all in separate subplots within the same figure.  Optionally
    saves the plots to files.

    Args:
        file_paths (list): A list of paths to the JSON files.
        save_plots (bool, optional): If True, saves the plots to PNG files.
                                     Defaults to True.
    """

    plt.figure(figsize=(12, 12))  # Adjust figure size as needed

    all_rewards = []
    all_steps = []
    num_episodes = None

    for file_path in file_paths:
        with open(file_path, 'r') as f:
            data = json.load(f)

        # Assuming each file has only one maze data
        maze_data = data[0]
        reward_history = maze_data['reward_history']
        steps_history = maze_data['steps_history']

        if num_episodes is None:
            num_episodes = len(reward_history)
        elif num_episodes != len(maze_data['reward_history']):
            raise ValueError(f"Inconsistent number of episodes in {file_path}")

        all_rewards.append(reward_history)
        all_steps.append(steps_history)

    all_rewards = np.array(all_rewards)  # Shape: (num_files, num_episodes)
    all_steps = np.array(all_steps)      # Shape: (num_files, num_episodes)

    episodes = range(1, num_episodes + 1)

    # Plot Reward History
    plt.subplot(2, 1, 1)  # 2 rows, 1 column, 1st subplot
    plt.title("Reward History per File with Average and Std Dev")
    plt.xlabel("Episode")
    plt.ylabel("Reward")

    for i in range(all_rewards.shape[0]):
        plt.plot(episodes, all_rewards[i, :], '.', alpha=0.5)  # Dot plot with transparency

    # Compute average and standard deviation across files for each episode
    avg_rewards = np.mean(all_rewards, axis=0)
    std_devs_rewards = np.std(all_rewards, axis=0)

    # Plot the overall average reward history as a line plot
    plt.plot(episodes, avg_rewards, color='blue', label='Average Reward')

    # Plot the standard deviation as a shaded region (error shading)
    plt.fill_between(episodes, avg_rewards - std_devs_rewards, avg_rewards + std_devs_rewards,
                     color='gray', alpha=0.2, label='Std Dev')

    plt.legend()
    plt.grid(True)

    if save_plots:
        plt.savefig("reward_history_plot.png")  # Save the reward subplot

    # Plot Step History
    plt.subplot(2, 1, 2)  # 2 rows, 1 column, 2nd subplot
    plt.title("Step History per File with Average and Std Dev")
    plt.xlabel("Episode")
    plt.ylabel("Steps")

    for i in range(all_steps.shape[0]):
        plt.plot(episodes, all_steps[i, :], '.', alpha=0.5)  # Dot plot with transparency

    # Compute average and standard deviation across files for each episode
    avg_steps = np.mean(all_steps, axis=0)
    std_devs_steps = np.std(all_steps, axis=0)

    # Plot the overall average steps history as a line plot
    plt.plot(episodes, avg_steps, color='green', label='Average Steps')

    # Plot the standard deviation as a shaded region (error shading)
    plt.fill_between(episodes, avg_steps - std_devs_steps, avg_steps + std_devs_steps,
                     color='orange', alpha=0.2, label='Std Dev')

    plt.legend()
    plt.grid(True)

    plt.tight_layout()

    if save_plots:
        plt.savefig("step_history_plot.png")  # Save the step subplot

    plt.show()  # Show the plots (if not saving)

# Create a list of file paths
file_paths = [f'./results/auto_gen_TD/ql_5_5_comp_{num}.json' for num in range(4)]  # Adjust range as needed

# Analyze and visualize the data
analyze_and_visualize_maze_data(file_paths, save_plots=True)