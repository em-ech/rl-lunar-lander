"""Minimal PyTorch DQN trainer for LunarLander (discrete actions).
This is a starter script: tweak hyperparameters and logging as needed.
"""
import argparse
import os
import time
import csv
import numpy as np
import gymnasium as gym
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt

from utils import ReplayBuffer, set_seed


class QNetwork(nn.Module):
    def __init__(self, obs_dim, n_actions):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, n_actions),
        )

    def forward(self, x):
        return self.net(x)


def train(args):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    env = gym.make(args.env)
    obs_dim = env.observation_space.shape[0]
    n_actions = env.action_space.n

    set_seed(args.seed)

    q = QNetwork(obs_dim, n_actions).to(device)
    q_target = QNetwork(obs_dim, n_actions).to(device)
    q_target.load_state_dict(q.state_dict())

    optimizer = optim.Adam(q.parameters(), lr=args.lr)
    replay = ReplayBuffer(args.replay_size)

    eps = args.eps_start
    episode_rewards = []
    os.makedirs(args.model_dir, exist_ok=True)
    os.makedirs(args.log_dir, exist_ok=True)
    os.makedirs(args.figures_dir, exist_ok=True)

    csv_path = os.path.join(args.log_dir, args.log_csv)
    # write header
    with open(csv_path, 'w', newline='') as fh:
        writer = csv.writer(fh)
        writer.writerow(['episode', 'reward', 'epsilon'])

    for ep in range(1, args.episodes + 1):
        obs, _ = env.reset()
        total_r = 0.0
        for t in range(args.max_steps):
            obs_v = torch.as_tensor(obs, dtype=torch.float32, device=device).unsqueeze(0)
            if np.random.rand() < eps:
                action = env.action_space.sample()
            else:
                with torch.no_grad():
                    qvals = q(obs_v)
                    action = int(torch.argmax(qvals, dim=-1).item())

            next_obs, reward, terminated, truncated, _ = env.step(action)
            done = bool(terminated or truncated)
            replay.push(obs, action, reward, next_obs, done)
            obs = next_obs
            total_r += reward

            if len(replay) >= args.batch_size:
                states, actions, rewards, next_states, dones = replay.sample(args.batch_size)
                states_v = torch.as_tensor(states, dtype=torch.float32, device=device)
                next_states_v = torch.as_tensor(next_states, dtype=torch.float32, device=device)
                actions_v = torch.as_tensor(actions, dtype=torch.long, device=device)
                rewards_v = torch.as_tensor(rewards, dtype=torch.float32, device=device)
                dones_v = torch.as_tensor(dones, dtype=torch.float32, device=device)

                q_vals = q(states_v)
                q_val = q_vals.gather(1, actions_v.unsqueeze(-1)).squeeze(-1)

                with torch.no_grad():
                    next_q = q_target(next_states_v)
                    max_next_q = next_q.max(1)[0]
                    target = rewards_v + args.gamma * max_next_q * (1.0 - dones_v)

                loss = nn.functional.mse_loss(q_val, target)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            if done:
                break

        # decay epsilon
        eps = max(args.eps_end, eps * args.eps_decay)

        # soft/hard update target
        if ep % args.target_update == 0:
            q_target.load_state_dict(q.state_dict())

        episode_rewards.append(total_r)
        # CSV logging
        with open(csv_path, 'a', newline='') as fh:
            writer = csv.writer(fh)
            writer.writerow([ep, float(total_r), float(eps)])

        if ep % args.log_interval == 0:
            avg = np.mean(episode_rewards[-args.log_interval:])
            print(f"Ep {ep}/{args.episodes}  avg_reward({args.log_interval}): {avg:.2f}  eps: {eps:.3f}")

        # periodic checkpoint + save rewards
        if ep % args.save_every == 0:
            ckpt_name = os.path.join(args.model_dir, f'dqn_ep{ep}.pt')
            torch.save(q.state_dict(), ckpt_name)
            np.save(os.path.join(args.log_dir, 'rewards.npy'), np.array(episode_rewards))
            print(f"Saved checkpoint: {ckpt_name}")

    # save final model and artifacts
    final_path = os.path.join(args.model_dir, 'dqn_final.pt')
    torch.save(q.state_dict(), final_path)
    np.save(os.path.join(args.log_dir, 'rewards.npy'), np.array(episode_rewards))

    # save a simple plot
    try:
        plt.figure()
        plt.plot(episode_rewards, label='episode reward')
        plt.xlabel('episode')
        plt.ylabel('reward')
        plt.legend()
        fig_path = os.path.join(args.figures_dir, 'training_rewards.png')
        plt.savefig(fig_path)
        plt.close()
        print(f"Saved final model: {final_path} and reward plot: {fig_path}")
    except Exception:
        print(f"Saved final model: {final_path}")


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--env', default='LunarLander-v3')
    p.add_argument('--episodes', type=int, default=200)
    p.add_argument('--max-steps', dest='max_steps', type=int, default=1200)
    p.add_argument('--lr', type=float, default=5e-4)
    p.add_argument('--gamma', type=float, default=0.99)
    p.add_argument('--batch-size', dest='batch_size', type=int, default=64)
    p.add_argument('--replay-size', dest='replay_size', type=int, default=100000)
    p.add_argument('--eps-start', type=float, default=1.0)
    p.add_argument('--eps-end', type=float, default=0.02)
    p.add_argument('--eps-decay', type=float, default=0.99)
    p.add_argument('--target-update', type=int, default=10)
    p.add_argument('--seed', type=int, default=0)
    p.add_argument('--model-dir', default='../models')
    p.add_argument('--log-dir', default='../logs')
    p.add_argument('--figures-dir', dest='figures_dir', default='../figures')
    p.add_argument('--save-every', dest='save_every', type=int, default=50)
    p.add_argument('--log-csv', dest='log_csv', default='training_log.csv')
    p.add_argument('--log-interval', type=int, default=10)
    return p.parse_args()


if __name__ == '__main__':
    args = parse_args()
    train(args)
