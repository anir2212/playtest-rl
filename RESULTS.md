# Measured results, 12 October 2026

All scores are from synthetic games created in this repository. Five independent training seeds per setting; intervals are percentile bootstrap intervals over **seed means** (2,000 resamples), not over individual episodes. Evaluation uses 30 Slicer or 100 GridQuest episodes per seed. These small games and training budgets do not imply performance in commercial titles. The exact seed rows and run configuration are committed in `evidence/`.

## Environment and algorithms

`gymnasium.utils.env_checker.check_env` passed for GridQuest and Slicer. GridQuest has a key, a locked door, coin, wall, hazard and goal. Slicer offers a six-value position/velocity/bomb observation and four actions. Nine tests passed, including replay-copy, target-terminal and GAE checks.

At **5,120 environment steps** per seed on Slicer:

| Agent | Mean fruit score / 80-step episode | 95% seed bootstrap CI | Mean bomb hits |
|---|---:|---:|---:|
| Scratch Double DQN | 3.513 | [2.260, 4.827] | 0.047 |
| SB3 DQN | 3.960 | [2.333, 5.600] | 0.060 |
| Scratch PPO | 1.800 | [1.553, 2.020] | 0.387 |
| SB3 PPO | 1.627 | [1.313, 1.887] | 0.340 |

DQN and PPO families received the same environment-step budget, and the PPO comparison uses four vector environments. Some implementation defaults differ (including termination handling), so these are reference checks, not matched benchmark claims. Wide DQN intervals mean the apparent 0.447 fruit difference is inconclusive. Under this short budget PPO learned less fruit score than DQN; more training and hyperparameter tuning are open.

PPO ablations at the same 5,120 steps: clipped reference mean fruit **1.800**, effectively unclipped policy **1.800**, and GAE λ=0 **1.787**. This bounded run shows no meaningful clipping/GAE effect; it does not establish those components are unnecessary.

## Reward design on GridQuest

With 500 training episodes per seed, Q-learning achieved **100%** held-out episode success for sparse, one-time shaped, count-bonus curiosity, and corrected one-time-key rewards, but **0%** for the intentionally farmable reward. The farmable version pays for each re-entry to the key cell, so the agent learns to loop and collect shaping reward instead of reaching the goal. The corrected reward pays only when the key is first acquired. All five seed values are identical at these endpoints; the trivial bootstrap intervals [1,1] or [0,0] reflect this small easy game, not general RL reliability. The two-by-two learning-rate/discount sweep is in `evidence/q.json`.

## Automated playtesting

At most 1,200 steps/seed were allowed; non-discovery is right-censored. A generic scripted bot plans for the key/goal and has no bug-location information. The online count-bonus Q policy receives an intrinsic novelty reward but is not pre-trained. The coverage policy is a state-action visitation heuristic. `unreachable_checkpoint` is detected at the first episode timeout (108 steps), so its equal times are a **test-oracle artifact**, not search superiority.

| Injected bug | Random | Scripted | Coverage heuristic | Count-bonus Q |
|---|---:|---:|---:|---:|
| Wall collision gap | 5/5, 49.2 steps | 0/5 | 5/5, 59.0 | 1/5, 473.0 |
| Score overflow | 5/5, 46.2 | 0/5 | 5/5, 35.4 | 5/5, 363.4 |
| Unreachable checkpoint | 5/5, 108.0 | 5/5, 108.0 | 5/5, 108.0 | 5/5, 108.0 |
| Softlock | 5/5, 181.6 | 0/5 | 5/5, 33.8 | 4/5, 577.3 |

Step means are calculated **only among discovered seeds** and should not be read as unconditional speed when detection counts differ. The coverage heuristic found all four bug types in five seeds, but was slower than random for the wall gap. The count-bonus Q policy underperformed in this small budget. `figures/coverage_heatmaps.png` shows visited positions for one seed, censored at discovery.

## Open work

Longer training curves and sample-efficiency thresholds, learned curiosity refinement, graphical pygame GIFs, richer difficulty sweep, independent bug test oracles, and an engine integration are pending. A browser or game-engine deployment was not made.
