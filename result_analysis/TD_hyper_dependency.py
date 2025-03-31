import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
import os
import json

# Collect all results into DataFrames
data = []
history_data = []
directory = "../experiments/results/auto_gen_TD"
for filename in os.listdir(directory):
    if (filename.startswith('sarsa_5x5_comp') or filename.startswith('ql_5x5_comp')) and filename.endswith('.json'):
        with open(os.path.join(directory, filename)) as f:
            entry = json.load(f)
            algorithm = 'SARSA' if filename.startswith('sarsa') else 'Q-Learning'
            
            # For aggregated performance metrics
            data.append({
                'algorithm': algorithm,
                'alpha': entry['alpha'],
                'gamma': entry['gamma'],
                'epsilon': entry['epsilon'],
                'complexity': entry['complexity'],
                'avg_reward': np.mean(entry['avg_rewards'][-100:]),
                'avg_steps': np.mean(entry['avg_steps'][-100:])
            })
            
            # For step history visualization
            for episode, steps in enumerate(entry['avg_steps']):
                history_data.append({
                    'algorithm': algorithm,
                    'complexity': entry['complexity'],
                    'episode': episode,
                    'steps': steps
                })

df = pd.DataFrame(data)
history_df = pd.DataFrame(history_data)

# Plot parameter comparison
plt.figure(figsize=(10, 15))
for idx, param in enumerate(['alpha', 'gamma', 'epsilon']):
    plt.subplot(3, 1, idx+1)
    param_df = df.groupby(['complexity', param, 'algorithm'])['avg_steps'].mean().reset_index()
    sns.lineplot(
        data=param_df, 
        x='complexity', 
        y='avg_steps', 
        hue='algorithm',
        style=param,
        markers=True,
        dashes=False,
        palette='colorblind'
    )
    plt.title(f'Impact of {param.capitalize()} on Performance')
    plt.xlabel('Maze Complexity')
    plt.ylabel('Average Steps (Last 100 Episodes)')
    plt.ylim(0, 1000)
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')

plt.tight_layout()

# Plot learning curves for step histories
plt.figure(figsize=(12, 8))
g = sns.FacetGrid(
    history_df,
    col='complexity',
    hue='algorithm',
    col_wrap=3,
    palette='colorblind',
    height=4,
    aspect=1.2
)
g.map(sns.lineplot, 'episode', 'steps', estimator='mean', errorbar=None)
g.set_axis_labels('Episode', 'Average Steps')
g.add_legend()
plt.suptitle('Learning Progress by Maze Complexity', y=1.02)
plt.tight_layout()

plt.show()
