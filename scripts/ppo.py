"""PPO trainer for LunarLander (PyTorch).
Starter implementation matching professor notebook defaults.
"""
import argparse
import os
import time
import csv
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import gymnasium as gym
import matplotlib.pyplot as plt

from utils import set_seed


class PolicyNetwork(nn.Module):
    def __init__(self, obs_dim, n_actions):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim, 64),
            nn.Tanh(),
            nn.Linear(64, 64),
            nn.Tanh(),
            nn.Linear(64, n_actions),
        )

    def forward(self, x):
        return self.net(x)


class ValueNetwork(nn.Module):
    def __init__(self, obs_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim, 64),
            nn.Tanh(),
            nn.Linear(64, 64),
            nn.Tanh(),
            nn.Linear(64, 1),
        )

    def forward(self, x):
        return self.net(x).squeeze(-1)


def compute_gae(rewards, values, dones, last_value, gamma, lam):
    T = len(rewards)
    advantages = np.zeros(T, dtype=np.float32)
    gae = 0.0
    for t in reversed(range(T)):
        if t == T - 1:
            next_value = last_value
        else:
            next_value = values[t + 1]
        mask = 1.0 - dones[t]
        delta = rewards[t] + gamma * next_value * mask - values[t]
        gae = delta + gamma * lam * mask * gae
        advantages[t] = gae
    returns = advantages + values
    return advantages, returns


def collect_trajectories(env, policy, value_net, steps, device, gamma, gae_lambda):
    obs_buf, act_buf, rew_buf, val_buf, logp_buf, done_buf = [], [], [], [], [], []
    obs, _ = env.reset()
    for _ in range(steps):
        obs_tensor = torch.as_tensor(obs, dtype=torch.float32, device=device).unsqueeze(0)
        with torch.no_grad():
            logits = policy(obs_tensor)
            value = value_net(obs_tensor).item()
            dist = torch.distributions.Categorical(logits=logits)
            action = dist.sample()
            logp = dist.log_prob(action).item()
            action = action.item()

        next_obs, reward, terminated, truncated, _ = env.step(action)
        obs_buf.append(obs)
        act_buf.append(action)
        rew_buf.append(reward)
        val_buf.append(value)
        logp_buf.append(logp)
        done_buf.append(1.0 if terminated else 0.0)

        obs = next_obs
        if terminated or truncated:
            obs, _ = env.reset()

    last_obs_tensor = torch.as_tensor(obs, dtype=torch.float32, device=device).unsqueeze(0)
    with torch.no_grad():
        last_value = value_net(last_obs_tensor).item()

    rewards = np.array(rew_buf, dtype=np.float32)
    values = np.array(val_buf, dtype=np.float32)
    dones = np.array(done_buf, dtype=np.float32)
    advantages, returns = compute_gae(rewards, values, dones, last_value, gamma, gae_lambda)

    return np.array(obs_buf, dtype=np.float32), np.array(act_buf, dtype=np.int64), advantages, returns, np.array(logp_buf, dtype=np.float32)


def ppo_update(policy, value_net, opt_policy, opt_value, obs, act, advantages, returns, old_logp, clip_ratio, epochs, batch_size, entropy_coef, max_grad_norm, device):
    advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
    obs_v = torch.as_tensor(obs, dtype=torch.float32, device=device)
    act_v = torch.as_tensor(act, dtype=torch.long, device=device)
    adv_v = torch.as_tensor(advantages, dtype=torch.float32, device=device)
    ret_v = torch.as_tensor(returns, dtype=torch.float32, device=device)
    old_logp_v = torch.as_tensor(old_logp, dtype=torch.float32, device=device)

    n = len(obs_v)
    for _ in range(epochs):
        idx = torch.randperm(n, device=device)
        for start in range(0, n, batch_size):
            batch_idx = idx[start:start + batch_size]
            logits = policy(obs_v[batch_idx])
            dist = torch.distributions.Categorical(logits=logits)
            new_logp = dist.log_prob(act_v[batch_idx])
            entropy = dist.entropy().mean()

            ratio = torch.exp(new_logp - old_logp_v[batch_idx])
            surr1 = ratio * adv_v[batch_idx]
            surr2 = torch.clamp(ratio, 1.0 - clip_ratio, 1.0 + clip_ratio) * adv_v[batch_idx]
            policy_loss = -torch.min(surr1, surr2).mean()

            loss_actor = policy_loss - entropy_coef * entropy
            opt_policy.zero_grad()
            loss_actor.backward()
            torch.nn.utils.clip_grad_norm_(policy.parameters(), max_grad_norm)
            opt_policy.step()

            # value update
            values_pred = value_net(obs_v[batch_idx])
            value_loss = ((values_pred - ret_v[batch_idx]) ** 2).mean()
            opt_value.zero_grad()
            value_loss.backward()
            torch.nn.utils.clip_grad_norm_(value_net.parameters(), max_grad_norm)
            opt_value.step()


def evaluate_policy(env, policy, device, n_episodes=10, deterministic=False):
    total_rewards = []
    for ep in range(n_episodes):
        obs, _ = env.reset(seed=10_000 + ep)
        done = False
        ep_reward = 0.0
        while not done:
            obs_tensor = torch.as_tensor(obs, dtype=torch.float32, device=device).unsqueeze(0)
            with torch.no_grad():
                logits = policy(obs_tensor)
                if deterministic:
                    action = torch.argmax(logits, dim=-1).item()
                else:
                    dist = torch.distributions.Categorical(logits=logits)
                    action = dist.sample().item()
            obs, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            ep_reward += reward
        total_rewards.append(ep_reward)
    return float(np.mean(total_rewards)), float(np.std(total_rewards))


def train(args):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    env = gym.make(args.env)
    obs_dim = env.observation_space.shape[0]
    n_actions = env.action_space.n

    set_seed(args.seed)

    policy = PolicyNetwork(obs_dim, n_actions).to(device)
    value_net = ValueNetwork(obs_dim).to(device)
    opt_policy = optim.Adam(policy.parameters(), lr=args.policy_lr)
    opt_value = optim.Adam(value_net.parameters(), lr=args.value_lr)

    os.makedirs(args.model_dir, exist_ok=True)
    os.makedirs(args.log_dir, exist_ok=True)
    os.makedirs(args.figures_dir, exist_ok=True)
    csv_path = os.path.join(args.log_dir, args.log_csv)
    with open(csv_path, 'w', newline='') as fh:
        writer = csv.writer(fh)
        writer.writerow(['iteration', 'mean_reward', 'std_reward'])

    avg_rewards = []
    start_time = time.time()
    for it in range(1, args.max_iters + 1):
        obs_b, act_b, adv_b, ret_b, logp_b = collect_trajectories(env, policy, value_net, args.steps_per_epoch, device, args.gamma, args.gae_lambda)
        ppo_update(policy, value_net, opt_policy, opt_value, obs_b, act_b, adv_b, ret_b, logp_b, args.clip_ratio, args.epochs, args.batch_size, args.entropy_coef, args.max_grad_norm, device)

        mean_r, std_r = evaluate_policy(env, policy, device, n_episodes=args.eval_eps, deterministic=False)
        avg_rewards.append(mean_r)
        with open(csv_path, 'a', newline='') as fh:
            writer = csv.writer(fh)
            writer.writerow([it, float(mean_r), float(std_r)])

        print(f"Iter {it}/{args.max_iters}  mean reward ({args.eval_eps} eps): {mean_r:.2f} ± {std_r:.2f}")

        if it % args.save_every == 0:
            torch.save(policy.state_dict(), os.path.join(args.model_dir, f'policy_it{it}.pt'))
            torch.save(value_net.state_dict(), os.path.join(args.model_dir, f'value_it{it}.pt'))

        if mean_r >= args.solved_threshold:
            print(f"Environment solved in {it} iterations")
            break

    # final save + plot
    torch.save(policy.state_dict(), os.path.join(args.model_dir, 'policy_final.pt'))
    torch.save(value_net.state_dict(), os.path.join(args.model_dir, 'value_final.pt'))
    np.save(os.path.join(args.log_dir, 'avg_rewards.npy'), np.array(avg_rewards))
    try:
        plt.figure()
        plt.plot(avg_rewards)
        plt.xlabel('iteration')
        plt.ylabel('mean eval reward')
        plt.savefig(os.path.join(args.figures_dir, 'ppo_eval_rewards.png'))
        plt.close()
    except Exception:
        pass


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--env', default='LunarLander-v3')
    p.add_argument('--policy-lr', type=float, default=3e-4)
    p.add_argument('--value-lr', type=float, default=1e-3)
    p.add_argument('--gamma', type=float, default=0.99)
    p.add_argument('--gae-lambda', dest='gae_lambda', type=float, default=0.95)
    p.add_argument('--clip-ratio', dest='clip_ratio', type=float, default=0.2)
    p.add_argument('--epochs', type=int, default=10)
    p.add_argument('--steps-per-epoch', dest='steps_per_epoch', type=int, default=2048)
    p.add_argument('--batch-size', type=int, default=64)
    p.add_argument('--entropy-coef', dest='entropy_coef', type=float, default=0.01)
    p.add_argument('--max-grad-norm', dest='max_grad_norm', type=float, default=0.5)
    p.add_argument('--max-iters', dest='max_iters', type=int, default=500)
    p.add_argument('--eval-eps', dest='eval_eps', type=int, default=10)
    p.add_argument('--model-dir', default='../models/ppo')
    p.add_argument('--log-dir', default='../logs/ppo')
    p.add_argument('--figures-dir', default='../figures/ppo')
    p.add_argument('--log-csv', dest='log_csv', default='ppo_log.csv')
    p.add_argument('--save-every', dest='save_every', type=int, default=10)
    p.add_argument('--seed', type=int, default=0)
    p.add_argument('--solved-threshold', type=float, default=200.0)
    return p.parse_args()


if __name__ == '__main__':
    args = parse_args()
    train(args)
"""Skeleton PPO trainer (PyTorch) — use professor notebook as reference to fill details.
This file is a starter and mirrors defaults used in the professor examples.
"""
import argparse

def main():
    print('This is a PPO trainer skeleton. Use the professor notebooks to fill in training details or use the provided `RL_course_prof/050_PPO_LunarLander_TORCH.ipynb` as a reference.')

if __name__ == '__main__':
    main()
