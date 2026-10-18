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

### Separate 10,240-step budget run, 18 October 2026

The same `run_neural` configuration was rerun from fresh seeded starts with **10,240 environment steps**, five training seeds (0–4), and 30 independent Slicer evaluation episodes per seed. The raw 20 agent/seed rows, settings and seed-level bootstrap summaries are in [`evidence/neural_10240.json`](evidence/neural_10240.json). This is an independent longer-budget endpoint run, **not a continuation** of the 5,120-step weights or a learning curve.

| Agent | Mean fruit score / episode | 95% seed bootstrap interval | Mean bomb hits |
|---|---:|---:|---:|
| Scratch Double DQN | 6.633 | [5.740, 7.187] | 0.047 |
| SB3 DQN | 6.173 | [5.300, 7.053] | 0.067 |
| Scratch PPO | 1.460 | [0.907, 1.860] | 0.253 |
| SB3 PPO | 1.767 | [1.607, 1.987] | 0.387 |

The DQN endpoints are higher than in the earlier 5,120-step run on this synthetic task, but the code scales its epsilon-decay schedule to the requested budget; SB3's exploration fraction is likewise defined against total steps. Changing the budget also changes the behavior schedule, so the difference is **not** an isolated effect of 5,120 extra transitions. Scratch and SB3 have other algorithm/default differences, and five seeds leave substantial uncertainty. PPO did not show the same clear endpoint increase under this setup. No steps-to-threshold or real-game performance claim follows from two separately trained endpoints.

PPO ablations at the same 5,120 steps: clipped reference mean fruit **1.800**, effectively unclipped policy **1.800**, and GAE λ=0 **1.787**. This bounded run shows no meaningful clipping/GAE effect; it does not establish those components are unnecessary.

## Reward design on GridQuest

With 500 training episodes per seed, Q-learning achieved **100%** held-out episode success for sparse, one-time shaped, count-bonus curiosity, and corrected one-time-key rewards, but **0%** for the intentionally farmable reward. The farmable version pays for each re-entry to the key cell, so the agent learns to loop and collect shaping reward instead of reaching the goal. The corrected reward pays only when the key is first acquired. All five seed values are identical at these endpoints; the trivial bootstrap intervals [1,1] or [0,0] reflect this small easy game, not general RL reliability. The two-by-two learning-rate/discount sweep is in `evidence/q.json`.

## Automated playtesting: original injected-label fixture (historical)

At most 1,200 steps/seed were allowed; non-discovery is right-censored. A generic scripted bot plans for the key/goal and has no bug-location information. The online count-bonus Q policy receives an intrinsic novelty reward but is not pre-trained. The coverage policy is a state-action visitation heuristic. `unreachable_checkpoint` is detected at the first episode timeout (108 steps), so its equal times are a **test-oracle artifact**, not search superiority.

This earlier fixture trusted `info["bug_detected"]`, and its score-overflow injection fired on the **first** coin visit despite the intended repeat-visit description. It is preserved for history, but the contract-based run below supersedes these detection numbers. Do not combine the two tables as if the oracle and environment were unchanged.

| Injected bug | Random | Scripted | Coverage heuristic | Count-bonus Q |
|---|---:|---:|---:|---:|
| Wall collision gap | 5/5, 49.2 steps | 0/5 | 5/5, 59.0 | 1/5, 473.0 |
| Score overflow | 5/5, 46.2 | 0/5 | 5/5, 35.4 | 5/5, 363.4 |
| Unreachable checkpoint | 5/5, 108.0 | 5/5, 108.0 | 5/5, 108.0 | 5/5, 108.0 |
| Softlock | 5/5, 181.6 | 0/5 | 5/5, 33.8 | 4/5, 577.3 |

Step means are calculated **only among discovered seeds** and should not be read as unconditional speed when detection counts differ. The coverage heuristic found all four bug types in five seeds, but was slower than random for the wall gap. The count-bonus Q policy underperformed in this small budget. `figures/coverage_heatmaps.png` shows visited positions for one seed, censored at discovery.

## Contract-based playtesting, 18 October 2026

The harness now checks observable GridQuest movement and score invariants instead of reading the injected `bug_detected` label. It flags an out-of-bounds position, score outside the declared 0–11 range, a blocked legal move, or a blocked goal entry. The score-overflow fixture was corrected to fire only when the coin tile is revisited. Five clean seeded traces of 300 actions each raised no contract violation; tests also drove each injected defect through a real transition. The new five-seed, 1,200-step-per-seed comparison and all 80 bug/policy/seed rows are in [`evidence/playtest_contract_oracle.json`](evidence/playtest_contract_oracle.json).

| Injected defect | Random | Scripted | Coverage heuristic | Count-bonus Q |
|---|---:|---:|---:|---:|
| Wall gap | 5/5, 49.2 steps | 0/5 | 5/5, 72.2 | 2/5, 972.5 |
| Score overflow on repeat visit | 5/5, 250.0 | 0/5 | 5/5, 58.0 | 5/5, 298.2 |
| Blocked goal entry | 5/5, 286.0 | 5/5, 10.0 | 5/5, 49.4 | 0/5 |
| Softlock blocking legal move | 5/5, 182.4 | 0/5 | 5/5, 35.4 | 5/5, 397.8 |

Each cell gives detections out of five seeds and the mean step **among detections only**. Misses are right-censored at 1,200 steps; for example, the count-bonus policy found no blocked goal in this run, so it has no mean discovery time. The evidence JSON also carries five-seed bootstrap intervals and coverage values. The scripted bot reaches the blocked goal in 10 steps because it follows a fixed key-to-goal route on this static map; this is an authored scenario, not general search performance. All detected labels matched the intended invariant class in this run. The oracle no longer reads the hidden bug label, but it still knows the game's declared map, score bound and movement rules. It has not been tested on unseen levels, game engines, or bugs outside these four injected modes.

## Open work

Checkpoint learning curves and sample-efficiency thresholds, learned curiosity refinement, graphical pygame GIFs, richer difficulty sweep, unseen-level and external-engine bug oracles, and an engine integration are pending. A browser or game-engine deployment was not made.
