# HW4 Quick Start

Reproduce the full submission on `LunarLander-v3`. Use the Python 3.11 `.venv`
(Box2D works there; the 3.14 `.venv-1` does not).

## One time setup

```bash
cd ~/Coding/Reinforcement\ Learning/hw4
python3.11 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
pip install "gymnasium[box2d]" imageio nbconvert ipykernel
python -m ipykernel install --user --name hw4-venv --display-name "Python (hw4 .venv)"
```

## Run the notebooks

Open in Jupyter and run top to bottom, in this order:

1. `notebooks/01_DQN_LunarLander.ipynb`
2. `notebooks/02_DDQN_LunarLander.ipynb`
3. `notebooks/03_PPO_LunarLander.ipynb`
4. `notebooks/05_execution_LunarLander.ipynb` (needs the three trained models)
5. `notebooks/04_compare_DQN_DDQN_PPO.ipynb` (needs logs, test results, and the sweep)

## Headless reproduce (command line)

```bash
cd ~/Coding/Reinforcement\ Learning/hw4
K="--ExecutePreprocessor.timeout=-1 --ExecutePreprocessor.kernel_name=hw4-venv"
for nb in 01_DQN 02_DDQN 03_PPO 05_execution_LunarLander; do
  .venv/bin/python -m nbconvert --to notebook --execute --inplace $K notebooks/${nb}*.ipynb
done
# PPO sensitivity sweep (Question 6), writes experiments/ppo_sensitivity.csv
( cd scripts && ../.venv/bin/python ppo_sensitivity.py )
# comparison last
.venv/bin/python -m nbconvert --to notebook --execute --inplace $K notebooks/04_compare_DQN_DDQN_PPO.ipynb
```

## What each run produces

- Models: `models/dqn/dqn_final.pt`, `models/ddqn/ddqn_final.pt`, `models/ppo/ppo_policy_final.pt`, `models/ppo/ppo_value_final.pt`
- Logs: `logs/{dqn,ddqn,ppo}/train_log.csv`
- Figures: training curves per agent, comparison plots, and one GIF per agent in `figures/`
- Experiments: `experiments/test_results.json`, `experiments/ppo_sensitivity.csv`

## Results summary

| Agent | Solved        | Env steps to solve | Test mean (30 ep) | Solved rate |
| ----- | ------------- | ------------------ | ----------------- | ----------- |
| DQN   | episode 270   | 144,933            | 239.2             | 80%         |
| DDQN  | episode 256   | 135,816            | 166.5             | 47%         |
| PPO   | iteration 110 | 225,280            | 215.7             | 80%         |

Full analysis and the seven discussion answers are in `RESULTS.md` and notebook 04.
