# Reinforcement Learning HW4: DQN, DDQN, and PPO on LunarLander

Train and compare three deep reinforcement learning agents on the discrete
`LunarLander-v3` environment: Deep Q Network (DQN), Double Deep Q Network (DDQN),
and Proximal Policy Optimization (PPO). All three are trained and evaluated on the
same environment, seed, and protocol. All three solve the task.

## Deliverables (notebooks)

The submission is driven by five notebooks in `notebooks/`, each executed with outputs saved.

| Notebook                         | Purpose                                                                                                                       |
| -------------------------------- | ----------------------------------------------------------------------------------------------------------------------------- |
| `01_DQN_LunarLander.ipynb`       | Train DQN until solved, save model, plot curves                                                                               |
| `02_DDQN_LunarLander.ipynb`      | Train DDQN (same network and schedule as DQN, double Q target), save, plot                                                    |
| `03_PPO_LunarLander.ipynb`       | Train PPO actor critic (GAE, clipped objective), save, plot                                                                   |
| `05_execution_LunarLander.ipynb` | Load each trained agent, run 30 test episodes, render one GIF per agent                                                       |
| `04_compare_DQN_DDQN_PPO.ipynb`  | Curves, sample efficiency, overestimation study, test table, PPO sensitivity, GIFs, and answers to the 7 discussion questions |

See `RESULTS.md` for the full results, tables, and discussion answers.

## Project structure

```
hw4/
  notebooks/     five deliverable notebooks (above), executed with outputs
  scripts/       supporting Python (training reference + ppo_sensitivity.py sweep)
  models/        trained weights: dqn/, ddqn/, ppo/
  logs/          per step training logs: dqn/, ddqn/, ppo/ (train_log.csv)
  figures/       training and comparison plots, one GIF per agent
  experiments/   test_results.json, ppo_sensitivity.csv
  requirements.txt
  RESULTS.md     results, tables, discussion answers
  QUICK_START.md how to reproduce
```

## Setup

Use the Python 3.11 virtual environment (`.venv`), which has Box2D working.

```bash
python3.11 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
pip install "gymnasium[box2d]" imageio nbconvert ipykernel
```

## Reproduce

From `notebooks/`, run the training notebooks first, then the execution notebook,
then the comparison notebook. See `QUICK_START.md` for one line commands including
headless execution.

## Results at a glance

| Agent | Solved        | Env steps to solve | Test mean (30 ep) | Test solved rate |
| ----- | ------------- | ------------------ | ----------------- | ---------------- |
| DQN   | episode 270   | 144,933            | 239.2             | 80%              |
| DDQN  | episode 256   | 135,816            | 166.5             | 47%              |
| PPO   | iteration 110 | 225,280            | 215.7             | 80%              |

Value based methods learn from fewer samples (DDQN most efficient, and the clearest
reduction in Q overestimation). PPO gives the smoothest training. On final held out
performance DQN and PPO are the strongest and most reliable.

## Acknowledgments

- Lunar Lander environment: Gymnasium by the Farama Foundation. https://gymnasium.farama.org/environments/box2d/lunar_lander/
- PPO algorithm: Schulman et al., Proximal Policy Optimization Algorithms, arXiv:1707.06347 (2017).
- Deep learning framework: PyTorch.
