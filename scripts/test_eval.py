"""Evaluate trained models on test set (30 deterministic episodes)."""
import argparse
import csv
import json
import os
from pathlib import Path
import numpy as np
import torch
import gymnasium as gym

from dqn import QNetwork as DQNNetwork
from ddqn import QNetwork as DDQNNetwork
from ppo import PolicyNetwork, ValueNetwork
from utils import set_seed


def eval_dqn_model(model_path, n_episodes=30, render=False):
    """Evaluate DQN model deterministically."""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    env = gym.make('LunarLander-v3', render_mode='rgb_array' if render else None)
    
    q = DQNNetwork(env.observation_space.shape[0], env.action_space.n).to(device)
    q.load_state_dict(torch.load(model_path, map_location=device))
    q.eval()
    
    rewards = []
    lengths = []
    solved_count = 0
    
    with torch.no_grad():
        for ep in range(n_episodes):
            obs, _ = env.reset()
            done = False
            total_r = 0
            steps = 0
            
            while not done and steps < 1200:
                obs_t = torch.as_tensor(obs, dtype=torch.float32, device=device).unsqueeze(0)
                action = q(obs_t).argmax(dim=1).item()
                obs, reward, terminated, truncated, _ = env.step(action)
                done = terminated or truncated
                total_r += reward
                steps += 1
            
            rewards.append(total_r)
            lengths.append(steps)
            if total_r >= 200:
                solved_count += 1
    
    env.close()
    return {
        'rewards': rewards,
        'mean': float(np.mean(rewards)),
        'std': float(np.std(rewards)),
        'max': float(np.max(rewards)),
        'min': float(np.min(rewards)),
        'final': float(rewards[-1]),
        'avg_length': float(np.mean(lengths)),
        'solved_rate': float(solved_count / n_episodes)
    }


def eval_ddqn_model(model_path, n_episodes=30, render=False):
    """Evaluate DDQN model deterministically."""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    env = gym.make('LunarLander-v3', render_mode='rgb_array' if render else None)
    
    q = DDQNNetwork(env.observation_space.shape[0], env.action_space.n).to(device)
    q.load_state_dict(torch.load(model_path, map_location=device))
    q.eval()
    
    rewards = []
    lengths = []
    solved_count = 0
    
    with torch.no_grad():
        for ep in range(n_episodes):
            obs, _ = env.reset()
            done = False
            total_r = 0
            steps = 0
            
            while not done and steps < 1200:
                obs_t = torch.as_tensor(obs, dtype=torch.float32, device=device).unsqueeze(0)
                action = q(obs_t).argmax(dim=1).item()
                obs, reward, terminated, truncated, _ = env.step(action)
                done = terminated or truncated
                total_r += reward
                steps += 1
            
            rewards.append(total_r)
            lengths.append(steps)
            if total_r >= 200:
                solved_count += 1
    
    env.close()
    return {
        'rewards': rewards,
        'mean': float(np.mean(rewards)),
        'std': float(np.std(rewards)),
        'max': float(np.max(rewards)),
        'min': float(np.min(rewards)),
        'final': float(rewards[-1]),
        'avg_length': float(np.mean(lengths)),
        'solved_rate': float(solved_count / n_episodes)
    }


def eval_ppo_model(policy_path, value_path, n_episodes=30, render=False):
    """Evaluate PPO model deterministically."""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    env = gym.make('LunarLander-v3', render_mode='rgb_array' if render else None)
    
    obs_dim = env.observation_space.shape[0]
    n_actions = env.action_space.n
    
    policy = PolicyNetwork(obs_dim, n_actions).to(device)
    policy.load_state_dict(torch.load(policy_path, map_location=device))
    policy.eval()
    
    rewards = []
    lengths = []
    solved_count = 0
    
    with torch.no_grad():
        for ep in range(n_episodes):
            obs, _ = env.reset()
            done = False
            total_r = 0
            steps = 0
            
            while not done and steps < 1200:
                obs_t = torch.as_tensor(obs, dtype=torch.float32, device=device).unsqueeze(0)
                logits = policy(obs_t)
                probs = torch.softmax(logits, dim=-1)
                action = probs.argmax(dim=1).item()  # Deterministic: take argmax
                obs, reward, terminated, truncated, _ = env.step(action)
                done = terminated or truncated
                total_r += reward
                steps += 1
            
            rewards.append(total_r)
            lengths.append(steps)
            if total_r >= 200:
                solved_count += 1
    
    env.close()
    return {
        'rewards': rewards,
        'mean': float(np.mean(rewards)),
        'std': float(np.std(rewards)),
        'max': float(np.max(rewards)),
        'min': float(np.min(rewards)),
        'final': float(rewards[-1]),
        'avg_length': float(np.mean(lengths)),
        'solved_rate': float(solved_count / n_episodes)
    }


def find_best_model(exp_dir, algo_name):
    """Find best performing model configuration."""
    import pandas as pd
    
    stats_file = exp_dir / 'statistics.csv'
    if not stats_file.exists():
        print(f"Statistics file not found: {stats_file}")
        return None
    
    df = pd.read_csv(stats_file)
    df_algo = df[df['exp'].str.startswith(algo_name)]
    
    if len(df_algo) == 0:
        print(f"No {algo_name} experiments found in statistics")
        return None
    
    best_idx = df_algo['mean'].idxmax()
    best_exp = df_algo.loc[best_idx, 'exp']
    print(f"Best {algo_name}: {best_exp} (mean={df_algo.loc[best_idx, 'mean']:.2f}, final={df_algo.loc[best_idx, 'final']:.2f})")
    
    return best_exp


def main():
    parser = argparse.ArgumentParser(description='Evaluate trained models')
    parser.add_argument('exp_dir', type=str, help='Experiment directory')
    parser.add_argument('--n-episodes', type=int, default=30, help='Number of test episodes')
    args = parser.parse_args()
    
    exp_dir = Path(args.exp_dir)
    set_seed(42)
    
    results = {}
    
    # Evaluate DQN
    print("\n=== DQN Evaluation ===")
    dqn_exp = find_best_model(exp_dir, 'dqn')
    if dqn_exp:
        dqn_dir = exp_dir / dqn_exp
        model_path = dqn_dir / 'models' / 'dqn_final.pt'
        if model_path.exists():
            results['dqn'] = eval_dqn_model(str(model_path), args.n_episodes)
            print(f"  Mean reward: {results['dqn']['mean']:.2f} ± {results['dqn']['std']:.2f}")
            print(f"  Solved rate: {results['dqn']['solved_rate']:.1%}")
        else:
            print(f"  Model not found: {model_path}")
    
    # Evaluate DDQN
    print("\n=== DDQN Evaluation ===")
    ddqn_exp = find_best_model(exp_dir, 'ddqn')
    if ddqn_exp:
        ddqn_dir = exp_dir / ddqn_exp
        model_path = ddqn_dir / 'models' / 'ddqn_final.pt'
        if model_path.exists():
            results['ddqn'] = eval_ddqn_model(str(model_path), args.n_episodes)
            print(f"  Mean reward: {results['ddqn']['mean']:.2f} ± {results['ddqn']['std']:.2f}")
            print(f"  Solved rate: {results['ddqn']['solved_rate']:.1%}")
        else:
            print(f"  Model not found: {model_path}")
    
    # Evaluate PPO
    print("\n=== PPO Evaluation ===")
    ppo_exp = find_best_model(exp_dir, 'ppo')
    if ppo_exp:
        ppo_dir = exp_dir / ppo_exp
        # Find the latest iteration
        policy_files = sorted(ppo_dir.glob('models/policy_it*.pt'))
        if policy_files:
            latest_it = policy_files[-1].stem.split('it')[-1]
            policy_path = ppo_dir / f'models/policy_it{latest_it}.pt'
            value_path = ppo_dir / f'models/value_it{latest_it}.pt'
            if policy_path.exists() and value_path.exists():
                results['ppo'] = eval_ppo_model(str(policy_path), str(value_path), args.n_episodes)
                print(f"  Mean reward: {results['ppo']['mean']:.2f} ± {results['ppo']['std']:.2f}")
                print(f"  Solved rate: {results['ppo']['solved_rate']:.1%}")
            else:
                print(f"  Models not found for iteration {latest_it}")
        else:
            print("  No PPO policy files found")
    
    # Save test results
    test_results_path = exp_dir / 'test_results.json'
    with open(test_results_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\nTest results saved to {test_results_path}")
    
    # Print summary
    print("\n=== Test Summary ===")
    for algo, stats in results.items():
        print(f"{algo.upper():5s}: mean={stats['mean']:7.2f}, solved={stats['solved_rate']:5.1%}, avg_len={stats['avg_length']:6.1f}")


if __name__ == '__main__':
    main()
