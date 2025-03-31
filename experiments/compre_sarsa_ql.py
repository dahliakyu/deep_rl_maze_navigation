import json
import os
import numpy as np
import pandas as pd
from scipy import stats
from statsmodels.stats.multitest import multipletests
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib
font = {'weight': 'bold', 'size': 22}

matplotlib.rc('font', **font)
# Configuration
MAZE_SIZE = "5x5"
COMPLEXITY_LEVELS = 15
AGENTS = ['ql', 'sarsa']
EPISODE_WINDOW = 100  # Last episodes to consider

def load_data(directory):
    """Load and process all JSON results files"""
    data = []
    
    for filename in os.listdir(directory):
        if not filename.endswith('.json'):
            continue
            
        # Parse filename components
        parts = filename.split('_')
        agent = parts[0]
        size_part = parts[1]
        comp = int(parts[3])
        
        if size_part != MAZE_SIZE:
            continue
            
        # Load JSON data
        with open(os.path.join(directory, filename)) as f:
            file_data = json.load(f)
            
        # Process rewards in each file and calculate average performance

        rewards = file_data['avg_rewards']
        avg_perf = np.mean(rewards[0:100])  # Last N episodes
        data.append({
            'agent': agent,
            'complexity': comp,
            'performance': avg_perf
        })
            
    return pd.DataFrame(data)

def calculate_effects(sarsa_data, ql_data):
    """Calculate statistical measures and effect sizes"""
    # Mann-Whitney U test
    mw_stat, mw_p = stats.mannwhitneyu(sarsa_data, ql_data)
    
    # Ensure we have valid sample sizes
    n1 = len(sarsa_data)
    n2 = len(ql_data)
    if n1 < 2 or n2 < 2:
        raise ValueError("Insufficient sample size for effect size calculation")
    
    # Proper pooled standard deviation
    var1 = np.std(sarsa_data, ddof=1)**2
    var2 = np.std(ql_data, ddof=1)**2
    pooled_var = ((n1 - 1)*var1 + (n2 - 1)*var2) / (n1 + n2 - 2)
    pooled_std = np.sqrt(pooled_var)
    
    # Cohen's d calculation
    cohen_d = (np.mean(sarsa_data) - np.mean(ql_data)) / pooled_std
    
    # Cliff's Delta
    def cliffs_delta(x, y):
        pairs = np.sum(np.greater.outer(x, y)) - np.sum(np.less.outer(x, y))
        return pairs / (len(x)*len(y))
    
    cd = cliffs_delta(sarsa_data, ql_data)
    
    return {
        'mw_p': mw_p,
        'cohen_d': cohen_d,
        'cliffs_delta': cd,
        'sarsa_mean': np.mean(sarsa_data),
        'ql_mean': np.mean(ql_data),
        'sarsa_ci': stats.t.interval(0.95, len(sarsa_data)-1, 
                                   loc=np.mean(sarsa_data), 
                                   scale=stats.sem(sarsa_data)),
        'ql_ci': stats.t.interval(0.95, len(ql_data)-1,
                                 loc=np.mean(ql_data), 
                                 scale=stats.sem(ql_data))
    }

def analyze_complexities(df):
    """Perform analysis across all complexity levels"""
    results = []
    
    for comp in range(COMPLEXITY_LEVELS):
        sarsa_data = df[(df['complexity'] == comp) & 
                       (df['agent'] == 'sarsa')]['performance'].values
        ql_data = df[(df['complexity'] == comp) & 
                    (df['agent'] == 'ql')]['performance'].values
        
        if len(sarsa_data) == 0 or len(ql_data) == 0:
            continue
            
        res = calculate_effects(sarsa_data, ql_data)
        res['complexity'] = comp
        results.append(res)
        
    results_df = pd.DataFrame(results)
    
    # Multiple comparisons correction
    pvals = results_df['mw_p'].values
    reject, pvals_corrected, _, _ = multipletests(pvals, alpha=0.05, method='holm')
    results_df['corrected_p'] = pvals_corrected
    results_df['significant'] = reject
    
    return results_df

def plot_results(results_df):
    """Visualize statistical comparison results"""
    plt.figure(figsize=(14, 7))
    
    # Plot means
    plt.plot(results_df['complexity'], results_df['sarsa_mean'], 
            label='SARSA', marker='o')
    plt.plot(results_df['complexity'], results_df['ql_mean'], 
            label='Q-Learning', marker='s')
    
    # Plot confidence intervals
    for comp, row in results_df.iterrows():
        plt.fill_between([row['complexity']-0.2, row['complexity']+0.2],
                        [row['sarsa_ci'][0], row['sarsa_ci'][0]],
                        [row['sarsa_ci'][1], row['sarsa_ci'][1]],
                        alpha=0.1, color='blue')
        plt.fill_between([row['complexity']-0.2, row['complexity']+0.2],
                        [row['ql_ci'][0], row['ql_ci'][0]],
                        [row['ql_ci'][1], row['ql_ci'][1]],
                        alpha=0.1, color='orange')
        
    # Add significance markers
    sig_points = results_df[results_df['significant']]
    for _, row in sig_points.iterrows():
        y_pos = max(row['sarsa_ci'][1], row['ql_ci'][1]) + 0.05
        plt.text(row['complexity'], y_pos, '*', ha='center', fontsize=14)
        
    plt.xlabel('Maze Complexity Level')
    plt.ylabel('Average Performance (First 100 Episodes)')
    plt.legend()
    plt.grid(alpha=0.2)
    plt.show()

def main():
    data_dir = "D./results/auto_gen_T"  # Update this path
    df = load_data(data_dir)
    results_df = analyze_complexities(df)
    
    # Print statistical results
    print("Statistical Results:")
    print(results_df[['complexity', 'sarsa_mean', 'ql_mean', 
                    'mw_p', 'corrected_p', 'significant',
                    'cohen_d', 'cliffs_delta']])
    
    # Generate visualization
    plot_results(results_df)

if __name__ == "__main__":
    main()