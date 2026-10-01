"""PPO sensitivity study for HW4 discussion Question 6.

Trains PPO on LunarLander-v3 while varying, one factor at a time, the three
hyperparameters the assignment asks about: the clip ratio (epsilon), the discount
factor (gamma), and the GAE parameter (lambda). Every run uses the SAME fixed
budget so the comparison is fair, and records how far each config gets.

Output: experiments/ppo_sensitivity.csv  (one row per config)
"""
import os, csv, time
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import gymnasium as gym

DEVICE = torch.device('cpu')
ENV_ID = 'LunarLander-v3'
SEED = 0

# fixed budget for every config (kept modest so the sweep runs in a few minutes)
ITERS         = 100
STEPS_PER_IT  = 2048
UPDATE_EPOCHS = 10
MINIBATCH     = 64
POLICY_LR     = 3e-4
VALUE_LR      = 1e-3
ENTROPY_COEF  = 0.01
MAX_GRAD_NORM = 0.5
EVAL_EVERY    = 10
EVAL_EPISODES = 20
SOLVED_SCORE  = 200.0

# base config + one-factor-at-a-time variations
BASE = dict(clip=0.2, gamma=0.99, lam=0.95)
GRID = (
    [dict(BASE, clip=c) for c in (0.1, 0.2, 0.3)] +
    [dict(BASE, gamma=g) for g in (0.98, 0.99, 0.999)] +
    [dict(BASE, lam=l)   for l in (0.90, 0.95, 0.97)]
)


class PolicyNetwork(nn.Module):
    def __init__(self, obs_dim, n_actions):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(obs_dim, 64), nn.Tanh(),
                                 nn.Linear(64, 64), nn.Tanh(),
                                 nn.Linear(64, n_actions))
    def forward(self, x):
        return self.net(x)


class ValueNetwork(nn.Module):
    def __init__(self, obs_dim):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(obs_dim, 64), nn.Tanh(),
                                 nn.Linear(64, 64), nn.Tanh(),
                                 nn.Linear(64, 1))
    def forward(self, x):
        return self.net(x).squeeze(-1)


def gae(rewards, values, dones, last_v, gamma, lam):
    T = len(rewards); adv = np.zeros(T, np.float32); g = 0.0
    for t in reversed(range(T)):
        nv = last_v if t == T - 1 else values[t + 1]
        mask = 1.0 - dones[t]
        delta = rewards[t] + gamma * nv * mask - values[t]
        g = delta + gamma * lam * mask * g
        adv[t] = g
    return adv, adv + values


def rollout(env, pol, val, steps, obs, gamma, lam):
    O, A, R, V, LP, D = [], [], [], [], [], []
    rets, cur = [], 0.0
    for _ in range(steps):
        ot = torch.as_tensor(obs, dtype=torch.float32, device=DEVICE).unsqueeze(0)
        with torch.no_grad():
            dist = torch.distributions.Categorical(logits=pol(ot)); v = val(ot).item()
            a = dist.sample(); lp = dist.log_prob(a).item(); a = a.item()
        nobs, r, term, trunc, _ = env.step(a)
        O.append(obs); A.append(a); R.append(r); V.append(v); LP.append(lp); D.append(1.0 if term else 0.0)
        obs = nobs; cur += r
        if term or trunc:
            rets.append(cur); cur = 0.0; obs, _ = env.reset()
    with torch.no_grad():
        last_v = val(torch.as_tensor(obs, dtype=torch.float32, device=DEVICE).unsqueeze(0)).item()
    R = np.array(R, np.float32); V = np.array(V, np.float32); D = np.array(D, np.float32)
    adv, ret = gae(R, V, D, last_v, gamma, lam)
    return (np.array(O, np.float32), np.array(A, np.int64), adv, ret, np.array(LP, np.float32)), obs, rets


def update(pol, val, op, ov, batch, clip):
    O, A, adv, ret, oldlp = batch
    adv = (adv - adv.mean()) / (adv.std() + 1e-8)
    O = torch.as_tensor(O, device=DEVICE); A = torch.as_tensor(A, device=DEVICE)
    adv = torch.as_tensor(adv, device=DEVICE); ret = torch.as_tensor(ret, device=DEVICE)
    oldlp = torch.as_tensor(oldlp, device=DEVICE); n = len(O)
    for _ in range(UPDATE_EPOCHS):
        idx = torch.randperm(n)
        for st in range(0, n, MINIBATCH):
            b = idx[st:st + MINIBATCH]
            dist = torch.distributions.Categorical(logits=pol(O[b]))
            newlp = dist.log_prob(A[b]); ent = dist.entropy().mean()
            ratio = torch.exp(newlp - oldlp[b])
            s1 = ratio * adv[b]; s2 = torch.clamp(ratio, 1 - clip, 1 + clip) * adv[b]
            ploss = -torch.min(s1, s2).mean() - ENTROPY_COEF * ent
            op.zero_grad(); ploss.backward(); nn.utils.clip_grad_norm_(pol.parameters(), MAX_GRAD_NORM); op.step()
            vloss = ((val(O[b]) - ret[b]) ** 2).mean()
            ov.zero_grad(); vloss.backward(); nn.utils.clip_grad_norm_(val.parameters(), MAX_GRAD_NORM); ov.step()


def evaluate(env, pol, n):
    rr = []
    for i in range(n):
        obs, _ = env.reset(seed=10_000 + i); done = False; R = 0.0
        while not done:
            ot = torch.as_tensor(obs, dtype=torch.float32, device=DEVICE).unsqueeze(0)
            with torch.no_grad():
                a = torch.argmax(pol(ot), dim=-1).item()
            obs, r, term, trunc, _ = env.step(a); done = term or trunc; R += r
        rr.append(R)
    return float(np.mean(rr)), float(np.std(rr))


def run_config(cfg):
    import random
    random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED)
    env = gym.make(ENV_ID); eval_env = gym.make(ENV_ID); env.reset(seed=SEED)
    od = env.observation_space.shape[0]; na = env.action_space.n
    pol = PolicyNetwork(od, na); val = ValueNetwork(od)
    op = optim.Adam(pol.parameters(), POLICY_LR); ov = optim.Adam(val.parameters(), VALUE_LR)
    obs, _ = env.reset(); best_eval = -1e9; solved_it = None; last_eval = float('nan')
    for it in range(1, ITERS + 1):
        batch, obs, _ = rollout(env, pol, val, STEPS_PER_IT, obs, cfg['gamma'], cfg['lam'])
        update(pol, val, op, ov, batch, cfg['clip'])
        if it % EVAL_EVERY == 0:
            m, _ = evaluate(eval_env, pol, EVAL_EPISODES)
            last_eval = m; best_eval = max(best_eval, m)
            if solved_it is None and m >= SOLVED_SCORE:
                solved_it = it
    env.close(); eval_env.close()
    return dict(best_eval=best_eval, final_eval=last_eval,
                solved_iter=solved_it if solved_it is not None else -1)


def main():
    out_dir = '../experiments'; os.makedirs(out_dir, exist_ok=True)
    out = os.path.join(out_dir, 'ppo_sensitivity.csv')
    rows = []
    t0 = time.time()
    for i, cfg in enumerate(GRID, 1):
        # label which factor is being varied relative to BASE
        varied = [k for k in ('clip', 'gamma', 'lam') if cfg[k] != BASE[k]]
        factor = varied[0] if varied else 'base'
        r = run_config(cfg)
        row = dict(factor=factor, clip=cfg['clip'], gamma=cfg['gamma'], lam=cfg['lam'], **r)
        rows.append(row)
        print(f"[{i}/{len(GRID)}] {factor:5s} clip={cfg['clip']} gamma={cfg['gamma']} lam={cfg['lam']} "
              f"-> best_eval={r['best_eval']:7.2f} final={r['final_eval']:7.2f} solved_iter={r['solved_iter']}")
    with open(out, 'w', newline='') as fh:
        w = csv.DictWriter(fh, fieldnames=['factor', 'clip', 'gamma', 'lam', 'best_eval', 'final_eval', 'solved_iter'])
        w.writeheader(); w.writerows(rows)
    print(f"\nWrote {out} | {len(rows)} configs | {time.time()-t0:.1f}s")


if __name__ == '__main__':
    main()
