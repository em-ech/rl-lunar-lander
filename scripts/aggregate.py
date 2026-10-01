"""Aggregates results from hyperparameter sweep runs and generates comparison plots.
"""
import os
import json
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path


def load_experiment_results(exp_dir):
    """Load CSV logs from an experiment directory."""
    logs = {}
    for algo_dir in exp_dir.iterdir():
        if not algo_dir.is_dir():
            continue
        for config_dir in algo_dir.iterdir():
            if not config_dir.is_dir():
                continue
            log_path = config_dir / 'logs' / '*.csv'
            csv_files = list(config_dir.glob('logs/*.csv'))
            for csv_file in csv_files:
                try:
                    df = pd.read_csv(csv_file)
                    key = f'{algo_dir.name}/{config_dir.name}'
                    logs[key] = df
                except Exception as e:
                    print(f'Failed to load {csv_file}: {e}')
    return logs


def compute_stats(logs):
    """Compute mean/std reward for each algorithm+config."""
    stats = []
    for key, df in logs.items():
        if 'reward' in df.columns:
            mean_reward = df['reward'].mean()
            std_reward = df['reward'].std()
            max_reward = df['reward'].max()
            final_reward = df['reward'].iloc[-1]
        elif 'mean_reward' in df.columns:
            mean_reward = df['mean_reward'].mean()
            std_reward = df['mean_reward'].std()
            max_reward = df['mean_reward'].max()
            final_reward = df['mean_reward'].iloc[-1]
        else:
            continue
        stats.append({
            'exp': key,
            'mean': mean_reward,
            'std': std_reward,
            'max': max_reward,
            'final': final_reward,
        })
    return pd.DataFrame(stats)


def plot_comparison(logs, out_dir):
    """Generate comparison plots."""
    plt.figure(figsize=(14, 8))
    colors = {'dqn': 'blue', 'ddqn': 'green', 'ppo': 'red'}
    
    for key, df in logs.items():
        algo = key.split('/')[0]
        color = colors.get(algo, 'black')
        reward_col = 'reward' if 'reward' in df.columns else 'mean_reward'
        plt.plot(df[reward_col], label=key, alpha=0.7, color=color)
    
    plt.xlabel('Episode / Iteration')
    plt.ylabel('Reward')
    plt.title('Training Curves Comparison')
    plt.legend(fontsize=8)
    plt.grid(alpha=0.3)
    plt.tight_layout()
    
    out_path = Path(out_dir) / 'comparison_curves.png'
    plt.savefig(out_path)
    print(f'Saved comparison plot to {out_path}')
    plt.close()


def main():
    import sys
    if len(sys.argv) < 2:
        print('Usage: python aggregate.py <experiments_dir>')
        sys.exit(1)
    
    exp_dir = Path(sys.argv[1])
    if not exp_dir.exists():
        print(f'Experiments directory not found: {exp_dir}')
        sys.exit(1)
    
    logs = load_experiment_results(exp_dir)
    stats = compute_stats(logs)
    
    print('\n=== Experiment Statistics ===')
    print(stats.to_string(index=False))
    
    plot_comparison(logs, exp_dir)
    
    # Save stats to CSV
    stats_path = exp_dir / 'statistics.csv'
    stats.to_csv(stats_path, index=False)
    print(f'\nStatistics saved to: {stats_path}')


if __name__ == '__main__':
    main()
