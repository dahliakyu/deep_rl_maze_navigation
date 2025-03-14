import json
import os
import re
import matplotlib.pyplot as plt
import numpy as np

def plot_hyperparameter_comparisons(directory='./'):
    # Collect all JSON files
    files = [f for f in os.listdir(directory) if f.endswith(".json")]
    
    # Regex pattern to extract hyperparameters
    pattern = r"all_episodes_lr_([0-9.]+)_gamma_([0-9.]+)_decay_([0-9.]+)\.json"
    
    # Data storage structure
    data = {}
    
    # Load and organize data
    for file in files:
        match = re.match(pattern, file)
        if not match:
            print(f"Skipping invalid filename: {file}")
            continue
            
        lr, gamma, decay = match.groups()
        with open(os.path.join(directory, file), 'r') as f:
            content = json.load(f)
            
        key = (gamma, decay)
        if key not in data:
            data[key] = {}
        data[key][lr] = {
            'steps': content['steps_history'],
            'rewards': content['reward_history']
        }

    # Create figures with subplots
    gammas = sorted({k[0] for k in data.keys()}, key=lambda x: float(x))
    decays = sorted({k[1] for k in data.keys()}, key=lambda x: float(x))
    
    fig_steps, axs_steps = plt.subplots(
        len(gammas), len(decays), 
        figsize=(20, 15), 
        squeeze=False
    )
    fig_steps.suptitle('Training Progress by Hyperparameters - Steps per Episode', fontsize=16)
    
    fig_rewards, axs_rewards = plt.subplots(
        len(gammas), len(decays), 
        figsize=(20, 15), 
        squeeze=False
    )
    fig_rewards.suptitle('Training Progress by Hyperparameters - Rewards per Episode', fontsize=16)

    # Plotting configuration
    lr_colors = {
        '0.0005': 'blue',
        '0.001': 'green',
        '0.002': 'red'
    }
    
    # Plot data in subplots
    for i, gamma in enumerate(gammas):
        for j, decay in enumerate(decays):
            key = (gamma, decay)
            if key not in data:
                continue
                
            ax_step = axs_steps[i][j]
            ax_reward = axs_rewards[i][j]
            
            lr_dict = data[key]
            for lr in sorted(lr_dict.keys(), key=lambda x: float(x)):
                steps = lr_dict[lr]['steps']
                rewards = lr_dict[lr]['rewards']
                episodes = np.arange(1, len(steps) + 1)
                
                # Apply smoothing
                window_size = 5
                smooth_steps = np.convolve(steps, np.ones(window_size)/window_size, mode='valid')
                smooth_rewards = np.convolve(rewards, np.ones(window_size)/window_size, mode='valid')
                smooth_episodes = episodes[window_size//2 : len(steps)-window_size//2]

                # Plot steps
                ax_step.plot(smooth_episodes, smooth_steps, 
                           color=lr_colors.get(lr, 'black'),
                           label=f'LR: {lr}')
                ax_step.set_title(f'γ: {gamma}, ε-decay: {decay}')
                ax_step.set_xlabel('Episodes')
                ax_step.set_ylabel('Steps (smoothed)')
                ax_step.grid(True)
                
                # Plot rewards
                ax_reward.plot(smooth_episodes, smooth_rewards,
                              color=lr_colors.get(lr, 'black'),
                              label=f'LR: {lr}')
                ax_reward.set_title(f'γ: {gamma}, ε-decay: {decay}')
                ax_reward.set_xlabel('Episodes')
                ax_reward.set_ylabel('Reward (smoothed)')
                ax_reward.grid(True)

            # Add legend to the first subplot in each row
            if j == 0:
                ax_step.legend(loc='upper right')
                ax_reward.legend(loc='lower right')

    plt.tight_layout()
    fig_steps.savefig('hyperparameter_steps_comparison.png', dpi=300, bbox_inches='tight')
    fig_rewards.savefig('hyperparameter_rewards_comparison.png', dpi=300, bbox_inches='tight')

if __name__ == "__main__":
    plot_hyperparameter_comparisons()