Experiment plan — DQN / DDQN / PPO

Baseline runs:

- DQN (PyTorch starter in `scripts/dqn.py`) with defaults: lr=5e-4, gamma=0.99, batch=64, replay=100k, eps_start=1.0 -> eps_end=0.02, eps_decay=0.99, target_update=10
- DDQN: follow professor implementation (`RL_course_prof/033_DDQN_LunarLander_TORCH.ipynb`) — same network sizes, soft/hard target update options
- PPO: use defaults from `RL_course_prof/050_PPO_LunarLander_TORCH.ipynb` (clip=0.2, gae_lambda=0.95, policy_lr=3e-4, value_lr=1e-3, steps_per_epoch=2048)

Small sweep suggestions (grid):

- DQN: lr ∈ {1e-3, 5e-4}, eps_decay ∈ {0.995, 0.99}, target_update ∈ {500, 2000}
- PPO: clip_ratio ∈ {0.1, 0.2}, gae_lambda ∈ {0.95, 0.97}, gamma ∈ {0.99, 0.995}

Evaluation protocol:

- 30 deterministic test episodes per saved model
- Metrics: mean ± std test reward, solved-rate (>=200), episode length distribution
- Plots: per-episode reward, rolling average (window=50), comparison figure with shaded std

Logging & artifacts:

- Save model checkpoints to `models/`
- Save training curves to `figures/` and logs to `logs/`
- Record one rendered episode per algorithm as GIF
