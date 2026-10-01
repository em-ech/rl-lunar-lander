"""Generate comprehensive comparison analysis and plots."""
import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

# Setup
sns.set_style('darkgrid')
exp_dir = Path('../experiments/20260611_144636')

print("=" * 80)
print("HW4 RL Algorithm Comparison: DQN vs DDQN vs PPO on LunarLander-v3")
print("=" * 80)

# Load statistics
stats_df = pd.read_csv(exp_dir / 'statistics.csv')
stats_df['algo'] = stats_df['exp'].str.split('/').str[0]
print("\n" + stats_df.to_string())

# Load test results
with open(exp_dir / 'test_results.json') as f:
    test_results = json.load(f)

print("\n" + "=" * 80)
print("TEST EVALUATION (30 deterministic episodes)")
print("=" * 80)
for algo, stats in test_results.items():
    print(f"\n{algo.upper()}:")
    print(f"  Mean reward: {stats['mean']:.2f} ± {stats['std']:.2f}")
    print(f"  Range: [{stats['min']:.2f}, {stats['max']:.2f}]")
    print(f"  Solved rate: {stats['solved_rate']:.1%} ({int(stats['solved_rate'] * 30)}/30)")
    print(f"  Avg episode length: {stats['avg_length']:.1f}")

# Analysis by algorithm
print("\n" + "=" * 80)
print("TRAINING PERFORMANCE ANALYSIS")
print("=" * 80)

for algo in ['dqn', 'ddqn', 'ppo']:
    algo_stats = stats_df[stats_df['algo'] == algo]
    if len(algo_stats) == 0:
        continue
    
    print(f"\n{algo.upper()}:")
    print(f"  Configurations: {len(algo_stats)}")
    print(f"  Mean reward range: [{algo_stats['mean'].min():.2f}, {algo_stats['mean'].max():.2f}]")
    print(f"  Std deviation range: [{algo_stats['std'].min():.2f}, {algo_stats['std'].max():.2f}]")
    
    best_idx = algo_stats['mean'].idxmax()
    best_config = algo_stats.loc[best_idx]
    print(f"  Best config: {best_config['exp']}")
    print(f"    Mean: {best_config['mean']:.2f}")
    print(f"    Max: {best_config['max']:.2f}")
    print(f"    Final: {best_config['final']:.2f}")

# Cross-algorithm comparison
print("\n" + "=" * 80)
print("CROSS-ALGORITHM COMPARISON")
print("=" * 80)

summary = []
for algo in ['dqn', 'ddqn', 'ppo']:
    algo_stats = stats_df[stats_df['algo'] == algo]
    if len(algo_stats) == 0:
        continue
    
    test_stat = test_results.get(algo, {})
    summary.append({
        'Algorithm': algo.upper(),
        'Configs': len(algo_stats),
        'Train Mean (best)': f"{algo_stats['mean'].max():.2f}",
        'Train Stability': f"{algo_stats['std'].mean():.2f}",
        'Test Mean': f"{test_stat.get('mean', 0):.2f}",
        'Test Solved': f"{test_stat.get('solved_rate', 0):.1%}",
    })

summary_df = pd.DataFrame(summary)
print("\n" + summary_df.to_string(index=False))

# Plot 1: Training mean vs std
print("\nGenerating comparison plots...")
fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# Training mean reward
ax = axes[0, 0]
for algo in ['dqn', 'ddqn', 'ppo']:
    algo_stats = stats_df[stats_df['algo'] == algo]
    if len(algo_stats) > 0:
        ax.scatter([algo.upper()] * len(algo_stats), algo_stats['mean'], s=100, alpha=0.6, label=algo)
        best_mean = algo_stats['mean'].max()
        ax.scatter([algo.upper()], [best_mean], s=200, marker='*', color='gold', edgecolor='black', linewidth=2)

ax.set_ylabel('Mean Reward (Training)')
ax.set_title('Training Performance Comparison')
ax.grid(alpha=0.3, axis='y')

# Training std (stability)
ax = axes[0, 1]
for algo in ['dqn', 'ddqn', 'ppo']:
    algo_stats = stats_df[stats_df['algo'] == algo]
    if len(algo_stats) > 0:
        ax.scatter([algo.upper()] * len(algo_stats), algo_stats['std'], s=100, alpha=0.6)
        mean_std = algo_stats['std'].mean()
        ax.scatter([algo.upper()], [mean_std], s=150, marker='D', color='red', edgecolor='black', linewidth=1.5)

ax.set_ylabel('Std Dev (Training)')
ax.set_title('Stability Comparison (lower is more stable)')
ax.grid(alpha=0.3, axis='y')

# Test evaluation
ax = axes[1, 0]
test_means = []
test_labels = []
for algo in ['dqn', 'ddqn', 'ppo']:
    if algo in test_results:
        test_means.append(test_results[algo]['mean'])
        test_labels.append(algo.upper())

ax.bar(test_labels, test_means, color=['blue', 'green', 'red'], alpha=0.6, edgecolor='black')
ax.axhline(y=200, color='gold', linestyle='--', linewidth=2, label='Solved (200)')
ax.set_ylabel('Mean Test Reward (30 episodes)')
ax.set_title('Test Evaluation Performance')
ax.legend()
ax.grid(alpha=0.3, axis='y')

# Solved rate
ax = axes[1, 1]
solved_rates = []
solved_labels = []
for algo in ['dqn', 'ddqn', 'ppo']:
    if algo in test_results:
        solved_rates.append(test_results[algo]['solved_rate'] * 100)
        solved_labels.append(algo.upper())

ax.bar(solved_labels, solved_rates, color=['blue', 'green', 'red'], alpha=0.6, edgecolor='black')
ax.set_ylabel('Solved Rate (%)')
ax.set_ylim([0, 100])
ax.set_title('Test Success Rate (reward ≥ 200)')
ax.grid(alpha=0.3, axis='y')

plt.tight_layout()
plt.savefig(exp_dir / 'algorithm_comparison.png', dpi=100, bbox_inches='tight')
print(f"Saved: {exp_dir}/algorithm_comparison.png")
plt.close()

# Plot 2: Training curves overlay (best of each algo)
fig, ax = plt.subplots(figsize=(12, 6))

for algo in ['dqn', 'ddqn', 'ppo']:
    algo_stats = stats_df[stats_df['algo'] == algo]
    if len(algo_stats) == 0:
        continue
    
    # Find best config
    best_idx = algo_stats['mean'].idxmax()
    best_exp = algo_stats.loc[best_idx, 'exp']
    
    # Load training log
    config_name = best_exp.split('/')[-1]
    log_path = exp_dir / best_exp / 'logs' / 'training_log.csv'
    
    if log_path.exists():
        df = pd.read_csv(log_path)
        if 'reward' in df.columns:
            rewards = df['reward'].values
        elif 'mean_reward' in df.columns:
            rewards = df['mean_reward'].values
        else:
            continue
        
        # Plot raw
        ax.plot(rewards, label=f'{algo.upper()} (raw)', alpha=0.3, linewidth=1)
        
        # Plot moving average
        ma = pd.Series(rewards).rolling(window=5, min_periods=1).mean()
        ax.plot(ma, label=f'{algo.upper()} (MA)', linewidth=2)

ax.set_xlabel('Episode / Iteration')
ax.set_ylabel('Reward')
ax.set_title('Best Configuration Comparison (Training Curves)')
ax.legend()
ax.grid(alpha=0.3)
plt.tight_layout()
plt.savefig(exp_dir / 'training_curves_best.png', dpi=100, bbox_inches='tight')
print(f"Saved: {exp_dir}/training_curves_best.png")
plt.close()

print("\n" + "=" * 80)
print("KEY INSIGHTS")
print("=" * 80)
print("""
1. **Training Speed**: Examine which algorithm reaches high rewards first
   - Check episode/iteration counts in moving average plots

2. **Stability**: Lower std deviation indicates more consistent learning
   - Smoother curves suggest better exploration/exploitation balance

3. **Test Performance**: Final test evaluation on 30 deterministic episodes
   - Solved rate shows percentage achieving 200+ reward threshold

4. **DDQN vs DQN**: DDQN decouples action selection from evaluation
   - Should reduce overestimation bias → different learning dynamics

5. **PPO Characteristics**: Actor-critic with clipped policy loss
   - On-policy: samples trajectories fresh each iteration
   - GAE for advantage estimation → smoother gradients

""")

print("Analysis complete! Check plots above and ", exp_dir / 'algorithm_comparison.png')
