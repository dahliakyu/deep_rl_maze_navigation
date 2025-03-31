import matplotlib
import matplotlib.pyplot as plt
import json

font = {'weight': 'bold', 'size': 22}

matplotlib.rc('font', **font)
# Configuration, change accordingly
files = ['../experiments/results/auto_gen_TD/sarsa_9x9_comp_15_a_0.1_g_0.9_e_0.1.json',  # SARSA file
         '..experiments/results/auto_gen_TD/ql_9x9_comp_15_a_0.1_g_0.9_e_0.1.json']    # Q-Learning file

# Create figure with twin axes
plt.figure(figsize=(10, 10))
ax = plt.gca()
ax2 = ax.twinx()

# Plot each agent's data
for file in files:
    with open(file) as f:
        data = json.load(f)
        agent = 'SARSA' if 'sarsa' in file else 'Q-Learning'
        color = 'tab:blue' if agent == 'SARSA' else 'tab:orange'
        
        # Plot steps (left axis)
        ax.plot(data['avg_steps'], 
                label=f'{agent} Steps', 
                color=color,
                linestyle='-')
        
        # Plot rewards (right axis)
        ax2.plot(data['avg_rewards'], 
                 label=f'{agent} Reward', 
                 color=color,
                 linestyle='--')

# Formatting
ax.set_xlabel('Training Episodes')
ax.set_ylabel('Average Steps', color='k')
ax2.set_ylabel('Average Reward', color='k')

# Combine legends
lines, labels = ax.get_legend_handles_labels()
lines2, labels2 = ax2.get_legend_handles_labels()
ax.legend(lines + lines2, labels + labels2, 
          loc='upper center', 
          bbox_to_anchor=(0.5, -0.15),
          ncol=2,
          fontsize=20)

plt.tight_layout()
plt.show()