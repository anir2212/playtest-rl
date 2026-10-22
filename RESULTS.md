# Measured results, 12–20 October 2026

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

### Checkpoint learning curves, 20 October 2026

This run trains **one model per agent and seed for 20,480 environment steps** and evaluates that same evolving network every 1,024 steps (21 checkpoints including the untrained step 0). It covers the two scratch agents, Double DQN and PPO, with training seeds 0–4. Each checkpoint uses the same 30 Slicer evaluation episodes (evaluation seed 1000 + training seed) with greedy actions. DQN's ε now decays over a fixed **3,584 absolute steps**, the same schedule the original 5,120-step run used, so a longer budget no longer stretches early exploration. PPO has no budget-dependent schedule. The protocol, including a **mean fruit score threshold of 5.0** per seed, was fixed in `analysis/curves.py` before the first run. All 210 checkpoint rows, parameter fingerprints and seed-level bootstrap summaries are in [`evidence/learning_curve_20480.json`](evidence/learning_curve_20480.json); the figure is `figures/learning_curve.png`. The five-seed run took about 33 seconds on this laptop. `make test` now passes 24 tests, 11 of them in `tests/test_curves.py`.

| Agent | Step 0 | 5,120 | 10,240 | 20,480 (95% seed bootstrap) | Bomb hits at 20,480 | Seeds reaching 5.0 |
|---|---:|---:|---:|---:|---:|---:|
| Scratch Double DQN | 0.773 | 3.513 | 5.920 | 8.047 [7.367, 8.727] | 0.513 | 5/5; first at 5,120–11,264 steps (median 8,192) |
| Scratch PPO | 0.293 | 1.800 | 1.460 | 2.367 [1.353, 3.913] | 0.187 | 1/5 (seed 3 at 20,480); 4 censored at 20,480 |

**Same-run check.** Evaluating a checkpoint does not change training: a test shows identical final weights with and without checkpoint evaluation, and each run's 21 parameter fingerprints differ from one another. As a stronger check, every seed's 5,120-step checkpoint equals the original, independently trained 5,120-step endpoint for both agents (DQN 3.513 and PPO 1.800, seed for seed). PPO's 10,240-step checkpoint also equals the separate 10,240-step endpoint run (1.460) seed for seed. DQN's does not (5.920 here against 6.633), as expected: that older run decayed ε over 7,168 steps rather than 3,584. The comparison is saved in [`evidence/learning_curve_consistency.json`](evidence/learning_curve_consistency.json).

**Protocol revision.** The first execution used 2,048-step spacing, which never lands on the 5,120-step comparison point. Spacing was changed to 1,024 and the run repeated, with budget, seeds, evaluation and threshold unchanged. The first run is kept as [`evidence/learning_curve_20480_every2048_superseded.json`](evidence/learning_curve_20480_every2048_superseded.json). At every checkpoint the two runs share, the parameter fingerprints and scores are identical.

**Limits.** On this synthetic task, DQN's score rose by about tenfold over 20,480 interactions, with dips between checkpoints (for example 3.840 at 4,096 and 3.513 at 5,120). Its bomb hits rose late in training, from 0.047 at 5,120 to 0.513 at 20,480. PPO stayed between about 0.9 and 2.4 after step 0, so this budget and these settings do not show PPO converging. The 21 points on a curve reuse the same 30 evaluation episodes, so they are repeated looks rather than independent tests. The threshold was set by this project, no checkpoint was selected as "best", and the curve covers one small game with one difficulty setting. SB3 references were not included in the curve. The result describes learning progress on Slicer; it does not show sample efficiency on another game.

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

## Held-out maps for the bug hunters, 20 October 2026 (evening)

**Protocol (frozen first).** `make unseen-protocol` wrote [`evidence/unseen_levels_protocol.json`](evidence/unseen_levels_protocol.json) before any hunt ran on these maps. It fixes:
- **Maps.** Eight maps from the seeded generator `unseen-v1` (seed 20261020; sizes 6, 7 and 8; manifest SHA-256 `56c38f87…`). Each map moves the start, key, coin, door, goal, walls, hazards, the wall-gap edge and the softlock trap. Every map is checked by breadth-first search to be playable, and the original map is excluded.
- **Search.** The same four policies and four defects as the original-map run, five hunt seeds per map, and the same 1,200-step budget.
- **Metrics.** Detections out of 40 (8 maps × 5 seeds) with a Wilson 95% interval, mean and median discovery step among detections only, and false alarms on a clean copy of every map. There was no tuning, and the comparison to the original map is descriptive only.

The contract oracle used each map's own walls, door, goal and size and never read the injected label. Before the protocol was written, the only checks on these maps were mechanics tests: each moved defect is observable, and 150-step clean random walks raise nothing. `make unseen` then ran once ([`evidence/unseen_levels_results.json`](evidence/unseen_levels_results.json)) in about 3 seconds.

| Injected defect | Random | Scripted | Coverage heuristic | Count-bonus Q |
|---|---:|---:|---:|---:|
| Wall gap | 38/40, 275.9 steps (original 5/5) | 0/40 (0/5) | 40/40, 73.5 (5/5) | 10/40, 450.4 (2/5) |
| Score overflow on repeat visit | 40/40, 137.8 (5/5) | 0/40 (0/5) | 40/40, 73.5 (5/5) | **12/40**, 595.8 (5/5) |
| Blocked goal entry | 39/40, 116.7 (5/5) | 40/40, 6.6 (5/5) | 40/40, 49.3 (5/5) | **26/40**, 300.7 (0/5) |
| Softlock blocking legal move | 40/40, 127.1 (5/5) | 5/40, 5.0 (0/5) | 40/40, 29.6 (5/5) | 25/40, 250.4 (5/5) |

Each cell gives detections out of 40 and the mean step among detections only. The original-map result is in brackets. The Wilson intervals, medians, per-map counts and violation kinds are in the evidence file.
- **No false alarms.** No policy raised a violation on any clean map: 0/40 runs each. On these maps the oracle flagged only injected defects.
- **Coverage and random transfer.** The coverage heuristic found every defect on every map and seed. Random found 157 of 160.
- **Scripted.** The key-then-goal bot still finds only the blocked goal. It found 5 softlocks, all on one map where the trap lay on its path, which it had never hit on the original map.
- **Count-bonus Q is map-sensitive.** Score overflow fell from 5/5 to 12/40, but the blocked goal rose from 0/5 to 26/40. A single development map gave a misleading picture of this policy in both directions.
- **One labelling detail.** One coverage softlock detection was labelled `goal_entry_blocked`, because the trap sat beside the goal. It is still a detection of the injected softlock.

**Limits.** All eight maps come from one small game and one generator, and the defect types are the same four. This tests location generalization, not new kinds of bug or another engine. The learning agents (Q, DQN, PPO) were not evaluated on unseen maps. Tabular Q's observation is a map-specific position index and cannot transfer by construction.

## Open work

Checkpoint curves for the two scratch Slicer agents are measured above. Still open: curves for the SB3 references and with more seeds, a difficulty sweep, unseen-level evaluation of the learning agents, new defect types and external-engine bug oracles, learned curiosity refinement, graphical pygame GIFs and an engine integration. A browser or game-engine deployment was not made.
