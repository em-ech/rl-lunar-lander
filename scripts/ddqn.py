"""DDQN trainer (PyTorch) for LunarLander. Implements soft/hard target updates and replay buffer.
"""
import argparse
import csv
import os
import time
import numpy as np
import random
import torch
import torch.nn as nn
import torch.optim as optim
import gymnasium as gym
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
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, n_actions),
        )

    def forward(self, x):
        return self.net(x)


def soft_update(target, source, tau):
    for t, s in zip(target.parameters(), source.parameters()):
        t.data.copy_(tau * s.data + (1.0 - tau) * t.data)


def train(args):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    env = gym.make(args.env)
    obs_dim = env.observation_space.shape[0]
    n_actions = env.action_space.n

    set_seed(args.seed)

    q = QNetwork(obs_dim, n_actions).to(device)
    q_target = QNetwork(obs_dim, n_actions).to(device)
    q_target.load_state_dict(q.state_dict())
    opt = optim.Adam(q.parameters(), lr=args.lr)
    replay = ReplayBuffer(args.replay_size)

    eps = args.eps_start
    rewards = []

    # setup logging
    os.makedirs(args.log_dir, exist_ok=True)
    csv_path = os.path.join(args.log_dir, 'training_log.csv')
    with open(csv_path, 'w', newline='') as fh:
        writer = csv.writer(fh)
        writer.writerow(['episode', 'reward', 'epsilon'])

    for ep in range(1, args.episodes + 1):
        obs, _ = env.reset()
        total_r = 0.0
        for t in range(args.max_steps):
            if random.random() < eps:
                action = env.action_space.sample()
            else:
                with torch.no_grad():
                    obs_v = torch.as_tensor(obs, dtype=torch.float32, device=device).unsqueeze(0)
                    action = int(torch.argmax(q(obs_v), dim=-1).item())

            next_obs, reward, terminated, truncated, _ = env.step(action)
            done = bool(terminated or truncated)
            replay.push(obs, action, reward, next_obs, done)
            obs = next_obs
            total_r += reward

            if len(replay) >= args.batch_size:
                states, actions, rewards_b, next_states, dones = replay.sample(args.batch_size)
                states_v = torch.as_tensor(states, dtype=torch.float32, device=device)
                next_states_v = torch.as_tensor(next_states, dtype=torch.float32, device=device)
                actions_v = torch.as_tensor(actions, dtype=torch.long, device=device)
                rewards_v = torch.as_tensor(rewards_b, dtype=torch.float32, device=device)
                dones_v = torch.as_tensor(dones, dtype=torch.float32, device=device)

                q_vals = q(states_v)
                q_val = q_vals.gather(1, actions_v.unsqueeze(-1)).squeeze(-1)

                with torch.no_grad():
                    next_actions = torch.argmax(q(next_states_v), dim=1)
                    next_qs = q_target(next_states_v)
                    next_q_val = next_qs[torch.arange(args.batch_size, device=device), next_actions]
                    target = rewards_v + args.gamma * next_q_val * (1.0 - dones_v)

                loss = nn.functional.mse_loss(q_val, target)
                opt.zero_grad()
                loss.backward()
                opt.step()

                # target network update
                if args.update == 'soft':
                    soft_update(q_target, q, args.tau)
                else:
                    if ep % args.target_update == 0:
                        q_target.load_state_dict(q.state_dict())

            if done:
                break

        eps = max(args.eps_end, eps * args.eps_decay)
        rewards.append(total_r)
        
        # CSV logging
        with open(csv_path, 'a', newline='') as fh:
            writer = csv.writer(fh)
            writer.writerow([ep, float(total_r), float(eps)])
        
        if ep % args.log_interval == 0:
            print(f"Ep {ep}/{args.episodes}  avg({args.log_interval}): {np.mean(rewards[-args.log_interval:]):.2f}  eps: {eps:.3f}")

        if ep % args.save_every == 0:
            os.makedirs(args.model_dir, exist_ok=True)
            torch.save(q.state_dict(), os.path.join(args.model_dir, f'ddqn_ep{ep}.pt'))
            np.save(os.path.join(args.log_dir, 'rewards.npy'), np.array(rewards))

    # final save
    os.makedirs(args.model_dir, exist_ok=True)
    torch.save(q.state_dict(), os.path.join(args.model_dir, 'ddqn_final.pt'))
    np.save(os.path.join(args.log_dir, 'rewards.npy'), np.array(rewards))
    try:
        plt.figure()
        plt.plot(rewards)
        plt.xlabel('episode')
        plt.ylabel('reward')
        plt.savefig(os.path.join(args.figures_dir, 'ddqn_rewards.png'))
        plt.close()
    except Exception:
        pass


def parse_args():
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument('--env', default='LunarLander-v3')
    p.add_argument('--episodes', type=int, default=500)
    p.add_argument('--max-steps', dest='max_steps', type=int, default=1200)
    p.add_argument('--lr', type=float, default=5e-4)
    p.add_argument('--gamma', type=float, default=0.99)
    p.add_argument('--batch-size', dest='batch_size', type=int, default=64)
    p.add_argument('--replay-size', dest='replay_size', type=int, default=100000)
    p.add_argument('--eps-start', type=float, default=1.0)
    p.add_argument('--eps-end', type=float, default=0.02)
    p.add_argument('--eps-decay', type=float, default=0.99)
    p.add_argument('--target-update', type=int, default=10)
    p.add_argument('--update', choices=['soft', 'hard'], default='soft')
    p.add_argument('--tau', type=float, default=0.005)
    p.add_argument('--seed', type=int, default=0)
    p.add_argument('--model-dir', default='../models/ddqn')
    p.add_argument('--log-dir', dest='log_dir', default='../logs/ddqn')
    p.add_argument('--figures-dir', dest='figures_dir', default='../figures/ddqn')
    p.add_argument('--save-every', dest='save_every', type=int, default=50)
    p.add_argument('--log-interval', type=int, default=10)
    return p.parse_args()


if __name__ == '__main__':
    args = parse_args()
    train(args)
