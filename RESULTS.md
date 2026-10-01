# HW4 Results: DQN vs DDQN vs PPO on LunarLander

## Setup

- Environment: `LunarLander-v3`, discrete 4 action space, 8 dimensional observation
- Same environment, seed, and evaluation protocol across all three agents
- Training target: solved threshold of mean return 200 over the last 100 episodes (value based) or a greedy 20 episode evaluation reaching 200 (PPO)
- Device: CPU (Apple M4 Pro; benchmarked faster than MPS for these small networks)
- Reproduced end to end by the five notebooks in `notebooks/`

All three agents solved the environment.

## Training outcome

| Agent | Solved at     | Final 100 window average | Environment steps to solve | Final mean predicted Q |
| ----- | ------------- | ------------------------ | -------------------------- | ---------------------- |
| DQN   | episode 270   | 200.35                   | 144,933                    | 31.13                  |
| DDQN  | episode 256   | 201.18                   | 135,816                    | 26.95                  |
| PPO   | iteration 110 | eval 234.54              | 225,280                    | not applicable         |

Sample efficiency: the replay based value methods reach the threshold in the fewest environment steps, DDQN the most efficient at about 136k steps versus 225k for on policy PPO.

Overestimation: DQN and DDQN are identical apart from the target computation, so the gap in the tracked mean predicted Q isolates the Double Q effect. DDQN ends about 4 points lower (26.95 versus 31.13), the expected signature of reduced overestimation bias.

## Test evaluation (30 deterministic episodes, identical seeds)

| Agent | Mean reward | Std   | Min | Max | Avg length | Solved | Solved rate |
| ----- | ----------- | ----- | --- | --- | ---------- | ------ | ----------- |
| DQN   | 239.2       | 48.6  | 108 | 301 | 430        | 24/30  | 80%         |
| DDQN  | 166.5       | 109.7 | -17 | 302 | 370        | 14/30  | 47%         |
| PPO   | 215.7       | 79.5  | -27 | 276 | 402        | 24/30  | 80%         |

On held out seeds DQN and PPO are the strongest and most reliable, both at 80 percent solved. The single DDQN checkpoint, although quickest to the training threshold and with lower predicted Q, generalized least well here and showed the highest variance. This is a reminder that the final saved weights are not always the best weights.

## PPO sensitivity study (Question 6)

One factor at a time from the base config (clip 0.2, gamma 0.99, lambda 0.95), fixed 100 iteration budget. `best_eval` is the best greedy 20 episode evaluation reached; `solved_iter` is the first iteration whose evaluation reached 200, with -1 meaning not reached within budget.

| Factor | Value       | best_eval | solved_iter                 |
| ------ | ----------- | --------- | --------------------------- |
| clip   | 0.1         | 186.0     | not reached                 |
| clip   | 0.2 (base)  | 171.9     | not reached                 |
| clip   | 0.3         | 276.1     | 50                          |
| gamma  | 0.98        | 200.7     | 50 (fell back, final 148.9) |
| gamma  | 0.99 (base) | 171.9     | not reached                 |
| gamma  | 0.999       | 272.9     | 50                          |
| lambda | 0.90        | 154.7     | not reached                 |
| lambda | 0.95 (base) | 171.9     | not reached                 |
| lambda | 0.97        | 225.6     | 70                          |

At this budget the more aggressive settings reach the threshold fastest. gamma is the most consequential factor: 0.999 was the best and most stable, while 0.98 hit 200 briefly then fell back. A wider clip (0.3) and a higher GAE lambda (0.97) also sped up learning on this seed. Caveat: single seed and a truncated budget, so these rankings measure speed to the threshold rather than asymptotic quality. The base config does solve given about 110 iterations (notebook 03).

## Answers to the discussion questions

1. Learning speed: value based methods learn from fewer environment steps. DDQN (about 136k) then DQN (about 145k) then PPO (about 225k).
2. Stability: PPO has the smoothest training curve, from the clipped objective and on policy batch averaging. DQN is the noisiest, DDQN sits between.
3. DDQN vs DQN overestimation: yes, reduced. Lower final predicted Q (about 27 versus 31) and slightly fewer steps to solve.
4. PPO smoother than DQN and DDQN: yes, clearly.
5. Best final evaluation reward: DQN (mean 239, 80 percent solved), PPO close behind (216, 80 percent solved). The final DDQN checkpoint was weaker here (167, 47 percent).
6. PPO sensitivity: see the study above. Sensitive to all three; gamma most consequential at this budget.
7. Trajectories: all three descend and fire to slow the touchdown. Value based agents make direct, decisive corrections; PPO looks smoother and more measured. Average test lengths differ (DQN 430, PPO 402, DDQN 370 steps).

## Takeaway

Off policy replay buys sample efficiency (DDQN the most efficient, and the clearest reduction in Q overestimation). On policy PPO buys training stability. On final held out performance DQN and PPO were the strongest and most reliable.

## Artifacts

- `notebooks/01_DQN_LunarLander.ipynb`, `02_DDQN_LunarLander.ipynb`, `03_PPO_LunarLander.ipynb`: training, executed with outputs
- `notebooks/05_execution_LunarLander.ipynb`: 30 episode test and GIFs
- `notebooks/04_compare_DQN_DDQN_PPO.ipynb`: curves, sample efficiency, overestimation, test table, sensitivity, GIFs, discussion answers
- `models/{dqn,ddqn,ppo}/`: trained weights
- `logs/{dqn,ddqn,ppo}/train_log.csv`: per step training logs
- `experiments/test_results.json`, `experiments/ppo_sensitivity.csv`: evaluation and sweep results
- `figures/`: training and comparison plots, plus one GIF per agent

## Acknowledgments

- Lunar Lander environment: Gymnasium by the Farama Foundation. https://gymnasium.farama.org/environments/box2d/lunar_lander/
- PPO algorithm: Schulman et al., Proximal Policy Optimization Algorithms, arXiv:1707.06347 (2017).
- Deep learning framework: PyTorch.
