# Measured results, 12 October–6 November 2026

All scores are from synthetic games created in this repository. Earlier runs use five independent training seeds per setting; the 2 November new-setting check and the 6 November GridQuest layout-transfer run use seven. Intervals are percentile bootstrap intervals over **seed means** (2,000 resamples), not over individual episodes; the layout-transfer run instead resamples layouts and seeds together with 5,000 resamples, as its own protocol declares. Evaluation uses 30 Slicer or 100 GridQuest episodes per seed, except in the layout-transfer run, which plays 5 sampled episodes (or 1 greedy episode) on each held-out layout per seed. These small games and training budgets do not imply performance in commercial titles. The exact seed rows and run configuration are committed in `evidence/`.

## Environment and algorithms

`gymnasium.utils.env_checker.check_env` passed for GridQuest and Slicer. GridQuest has a key, a locked door, coin, wall, hazard and goal. Slicer offers a six-value position/velocity/bomb observation and four actions. At that first run (12 October), nine tests passed, including replay-copy, target-terminal and GAE checks.

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

This run trains **one model per agent and seed for 20,480 environment steps** and evaluates that same evolving network every 1,024 steps (21 checkpoints including the untrained step 0). It covers the two scratch agents, Double DQN and PPO, with training seeds 0–4. Each checkpoint uses the same 30 Slicer evaluation episodes (evaluation seed 1000 + training seed) with greedy actions. DQN's ε now decays over a fixed **3,584 absolute steps**, the same schedule the original 5,120-step run used, so a longer budget no longer stretches early exploration. PPO has no budget-dependent schedule. The protocol, including a **mean fruit score threshold of 5.0** per seed, was fixed in `analysis/curves.py` before the first run. All 210 checkpoint rows, parameter fingerprints and seed-level bootstrap summaries are in [`evidence/learning_curve_20480.json`](evidence/learning_curve_20480.json); the figure is `figures/learning_curve.png`. The five-seed run took about 33 seconds on this laptop. At that point `make test` passed 24 tests, 11 of them in `tests/test_curves.py`.

| Agent | Step 0 | 5,120 | 10,240 | 20,480 (95% seed bootstrap) | Bomb hits at 20,480 | Seeds reaching 5.0 |
|---|---:|---:|---:|---:|---:|---:|
| Scratch Double DQN | 0.773 | 3.513 | 5.920 | 8.047 [7.367, 8.727] | 0.513 | 5/5; first at 5,120–11,264 steps (median 8,192) |
| Scratch PPO | 0.293 | 1.800 | 1.460 | 2.367 [1.353, 3.913] | 0.187 | 1/5 (seed 3 at 20,480); 4 censored at 20,480 |

**Same-run check.** Evaluating a checkpoint does not change training: a test shows identical final weights with and without checkpoint evaluation, and each run's 21 parameter fingerprints differ from one another. As a stronger check, every seed's 5,120-step checkpoint equals the original, independently trained 5,120-step endpoint for both agents (DQN 3.513 and PPO 1.800, seed for seed). PPO's 10,240-step checkpoint also equals the separate 10,240-step endpoint run (1.460) seed for seed. DQN's does not (5.920 here against 6.633), as expected: that older run decayed ε over 7,168 steps rather than 3,584. The comparison is saved in [`evidence/learning_curve_consistency.json`](evidence/learning_curve_consistency.json).

**Protocol revision.** The first execution used 2,048-step spacing, which never lands on the 5,120-step comparison point. Spacing was changed to 1,024 and the run repeated, with budget, seeds, evaluation and threshold unchanged. The first run is kept as [`evidence/learning_curve_20480_every2048_superseded.json`](evidence/learning_curve_20480_every2048_superseded.json). At every checkpoint the two runs share, the parameter fingerprints and scores are identical.

**Limits.** On this synthetic task, DQN's score rose by about tenfold over 20,480 interactions, with dips between checkpoints (for example 3.840 at 4,096 and 3.513 at 5,120). Its bomb hits rose late in training, from 0.047 at 5,120 to 0.513 at 20,480. PPO stayed between about 0.9 and 2.4 after step 0, so this budget and these settings do not show PPO converging. The 21 points on a curve reuse the same 30 evaluation episodes, so they are repeated looks rather than independent tests. The threshold was set by this project, no checkpoint was selected as "best", and the curve covers one small game with one difficulty setting (held-out settings were evaluated later; see the 22 October section). SB3 references were not included in the curve; their curve was measured on 23 October under a separate protocol (see that section). The result describes learning progress on Slicer; it does not show sample efficiency on another game.

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

Each cell gives detections out of five seeds and the mean step **among detections only**. Misses are right-censored at 1,200 steps; for example, the count-bonus policy found no blocked goal in this run, so it has no mean discovery time. The evidence JSON also carries five-seed bootstrap intervals and coverage values. The scripted bot reaches the blocked goal in 10 steps because it follows a fixed key-to-goal route on this static map; this is an authored scenario, not general search performance. All detected labels matched the intended invariant class in this run. The oracle no longer reads the hidden bug label, but it still knows the game's declared map, score bound and movement rules. As of 18 October it had not been tested on unseen levels; the 20 October held-out map run below covers that. It has not been tested on game engines. As of 18 October it had not met a bug outside these four injected modes; the 23 October section below adds one, which this contract misses entirely.

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

**Limits.** All eight maps come from one small game and one generator, and the defect types are the same four. This tests location generalization, not new kinds of bug or another engine (one new defect type was added on 23 October under its own protocol; see below). As of 20 October the learning agents (Q, DQN, PPO) had not been evaluated on unseen maps. Tabular Q's observation is a map-specific position index and cannot transfer by construction. The neural agents were first evaluated on held-out Slicer settings instead (22 October section); a PPO agent with a transferable GridQuest view was evaluated on held-out layouts on 6 November (see that section).

## Held-out Slicer difficulty settings for the learning agents, 22 October 2026

**Protocol (committed first).** `make difficulty-protocol` wrote [`evidence/slicer_difficulty_protocol.json`](evidence/slicer_difficulty_protocol.json) from the `PROTOCOL` constant in `analysis/difficulty.py`. It was committed on its own (`f3f175d`) before any agent was trained or evaluated on a held-out setting. Before that commit, the only runs touching the new code were unit tests and smoke runs on the default setting and on a test-only setting (bomb share 0.1) that is not in the protocol. The protocol fixes:
- **Settings.** The default Slicer (bomb share 0.2, speed 1.0) is the only training setting. Seven settings are held out: `no_bombs` (0.0, 1.0), `slow_0.7` (0.2, 0.7), `bombs_0.4` (0.4, 1.0), `bombs_0.6` (0.6, 1.0), `fast_1.3` (0.2, 1.3), `fast_1.6` (0.2, 1.6) and `bombs_0.4_fast_1.3` (0.4, 1.3). `speed` is a new `Slicer` argument that scales both sampled object velocities. Its default of 1.0 leaves every trajectory unchanged, and the curve check below confirms this. Horizon, cursor step, slice window and rewards are unchanged. Every setting stays visible in the six-value observation through the bomb flag and the velocities.
- **Agents and training.** Scratch Double DQN and scratch PPO with training seeds 0–4 (the code asserts five distinct seeds per setting). Training uses the frozen curve protocol: 20,480 steps, checkpoints every 1,024 steps, DQN ε from 1.0 to 0.05 over 3,584 absolute steps, PPO 512-step rollouts. Nothing was tuned. Tabular Q is excluded because its state cannot transfer. SB3 is excluded because it had no curve protocol at the time (one was added on 23 October).
- **Evaluation.** 30 greedy episodes per seed and setting. Evaluation seed 7000 + training seed is a block no earlier run used, so episode *e* resets with seed (7000 + training seed) × 100,000 + *e*. The default-setting models are also evaluated with the curve's seeds (1000 + training seed), for the consistency check only.
- **References.** For each held-out setting, fresh agents with the same seeds and budget are trained directly on it: the in-setting reference, which forms the difficulty sweep. Two non-learning anchors play the same episodes: a uniform random player, and a fixed scripted rule written before the run (stay on a bomb; on a fruit, slice when it is in reach next tick, otherwise step toward it).
- **Metrics.** Primary: per-seed mean fruit score at 20,480 steps, five-seed mean with the seed-level bootstrap interval, with mean bomb hits reported alongside. Secondary: an anchor-normalized score, (agent − random) / (scripted − random). Also the transfer gap, defined as in-setting reference mean minus transferred mean, with a bootstrap over the two independent five-seed groups. Its reading is "transfer loss" if the 95% interval is above 0, "transfer gain" if below 0, and otherwise "no measurable difference". Held-out score at every checkpoint is descriptive only.

`make difficulty` then ran once, with four worker processes, in about two minutes. Its output is committed as [`evidence/slicer_difficulty_20480.json`](evidence/slicer_difficulty_20480.json) (a compact copy of `runs/difficulty.json`), and `make difficulty-figure` draws `figures/slicer_difficulty.png` from it. The run comprised 80 training runs: 2 agents × 5 seeds on the default setting, plus 2 agents × 5 seeds on each of the 7 held-out settings. Each model was evaluated at 21 checkpoints. The anchors played 8 settings × 5 evaluation seeds × 30 episodes each.

**Curve check.** All **210 of 210** default-setting checkpoints evaluated with the curve's seeds matched `evidence/learning_curve_20480.json`: score, bomb hits, return and parameter fingerprint. The new speed argument and the multi-setting evaluation therefore left training unchanged.

**Reproduction and older checks.** A second `make difficulty` afterwards reproduced every training row and anchor in the committed evidence (`matches_committed_evidence: true`). After the run, `make curve` still reported `matches_protocol: true` with rows identical to `evidence/learning_curve_20480.json`. `.venv/bin/python -m analysis.curves` still showed identical 5,120-step checks for both agents, and its consistency file was unchanged. `make unseen` regenerated the frozen maps and reproduced the held-out bug-hunt rows and table exactly. `make test` passes 51 tests: 12 in `tests/test_difficulty.py` and one new Slicer speed test.

Each cell below is a five-seed mean of 30-episode means (5 × 30 = 150 episodes) at 20,480 steps, with the 95% seed bootstrap interval. The anchors use the same 150 episodes. "Normalized" is the five-seed mean anchor-normalized score; its intervals are in the evidence file. The gap is the in-setting reference minus the model trained on the default setting.

**Scratch Double DQN**

| Setting (bomb share, speed) | Random | Scripted | Trained on default: fruit | Bomb hits | Normalized | Trained on this setting: fruit | Bomb hits | Gap and reading |
|---|---:|---:|---:|---:|---:|---:|---:|---|
| train (0.2, 1.0) | 1.513 | 7.127 | 7.680 [7.387, 7.973] | 0.500 | 1.10 | (same models) | | |
| no_bombs (0.0, 1.0) | 1.953 | 14.287 | 14.087 [13.873, 14.267] | 0.000 | 0.98 | 14.133 [14.027, 14.253] | 0.000 | 0.047 [−0.167, 0.280], no measurable difference |
| slow_0.7 (0.2, 0.7) | 1.233 | 5.307 | 5.820 [5.667, 6.067] | 0.433 | 1.13 | 5.107 [3.880, 5.920] | 0.367 | −0.713 [−1.933, 0.147], no measurable difference |
| bombs_0.4 (0.4, 1.0) | 1.147 | 3.820 | 4.333 [4.160, 4.500] | 0.767 | 1.21 | 3.653 [3.267, 4.053] | 0.000 | −0.680 [−1.133, −0.240], transfer gain |
| bombs_0.6 (0.6, 1.0) | 0.787 | 1.920 | 2.293 [2.147, 2.440] | 0.973 | 1.36 | 1.613 [1.306, 1.833] | 0.000 | −0.680 [−1.027, −0.393], transfer gain |
| fast_1.3 (0.2, 1.3) | 1.593 | 8.867 | 9.540 [9.247, 9.867] | 0.653 | 1.10 | 9.680 [8.640, 10.940] | 0.847 | 0.140 [−0.987, 1.414], no measurable difference |
| fast_1.6 (0.2, 1.6) | 1.860 | 10.267 | 10.927 [10.613, 11.280] | 0.760 | 1.08 | 10.680 [9.207, 12.200] | 0.547 | −0.247 [−1.713, 1.233], no measurable difference |
| bombs_0.4_fast_1.3 (0.4, 1.3) | 1.207 | 4.813 | 5.420 [5.147, 5.667] | 0.933 | 1.19 | 4.607 [4.167, 5.027] | 0.033 | −0.813 [−1.347, −0.340], transfer gain |

**Scratch PPO** (same anchors)

| Setting | Trained on default: fruit | Bomb hits | Normalized | Trained on this setting: fruit | Bomb hits | Gap and reading |
|---|---:|---:|---:|---:|---:|---|
| train | 2.127 [1.460, 3.240] | 0.213 | 0.11 | (same models) | | |
| no_bombs | 3.540 [2.107, 5.980] | 0.000 | 0.13 | 5.087 [2.819, 7.747] | 0.000 | 1.547 [−1.747, 4.841], no measurable difference |
| slow_0.7 | 1.453 [0.960, 2.193] | 0.140 | 0.06 | 1.380 [1.073, 1.687] | 0.153 | −0.073 [−0.873, 0.540], no measurable difference |
| bombs_0.4 | 1.347 [0.980, 1.880] | 0.493 | 0.08 | 0.707 [0.607, 0.807] | 0.040 | −0.640 [−1.153, −0.260], transfer gain |
| bombs_0.6 | 0.820 [0.600, 1.107] | 0.660 | 0.01 | 0.560 [0.347, 0.833] | 0.000 | −0.260 [−0.620, 0.080], no measurable difference |
| fast_1.3 | 2.713 [1.940, 4.027] | 0.233 | 0.15 | 3.027 [2.287, 4.113] | 0.353 | 0.313 [−1.167, 1.753], no measurable difference |
| fast_1.6 | 3.127 [2.280, 4.520] | 0.280 | 0.15 | 2.400 [2.040, 2.800] | 0.400 | −0.727 [−2.180, 0.267], no measurable difference |
| bombs_0.4_fast_1.3 | 1.680 [1.227, 2.260] | 0.587 | 0.12 | 1.073 [0.727, 1.660] | 0.000 | −0.607 [−1.320, 0.133], no measurable difference |

The scripted anchor hit no bombs on any setting.
- **No transfer loss.** Across the 14 agent-and-setting pairs, no gap interval lies above zero. For DQN, 4 of 7 readings are "no measurable difference" and 3 of 7 are "transfer gain". For PPO, 6 of 7 are "no measurable difference" and 1 of 7 is "transfer gain". With five seeds per group, "no measurable difference" is not evidence that the two are equal. The PPO and slower or faster DQN intervals are wide.
- **DQN held up on faster and slower objects.** That includes `fast_1.6`, whose vertical speeds (0.048–0.088 per tick) lie partly outside anything seen in training (0.03–0.055). Its fruit score beat the scripted rule on every setting except `no_bombs`, where both are near the ceiling (14.087 against 14.287).
- **The "transfer gains" are on fruit score only.** On the three bomb-heavy settings, the model trained on the default setting sliced more fruit than the in-setting reference. It also hit 0.77–0.97 bombs per episode, against 0.00–0.03 for the references. The following check was not pre-registered, and no interval is attached. Mean episode return in the evidence file counts −1 per bomb, −0.1 per missed fruit and a small tick cost. On that return, the default-trained DQN scored slightly below the references on all three bomb-heavy settings: 3.192 against 3.270, 0.936 against 1.212, and 4.118 against 4.193. The scripted rule's returns were higher than both: 3.439, 1.530 and 4.437. The references appear to have learned cautious, bomb-avoiding play within 20,480 steps rather than less skill overall. The fruit-score reading should not be presented as an outright advantage.
- **PPO had little to transfer.** Its normalized score stayed between 0.01 and 0.15. On the training setting the interval for that score, [−0.014, 0.310], includes zero, which means PPO was barely better than random. Its gaps therefore say little about generalization. Seed 3 is the strongest PPO seed on every setting; for example, it scored 4.267 on the training setting.
- **Held-out score over training (descriptive).** For DQN, held-out scores rose alongside the training-setting score. At 20,480 steps every setting was within 0.4 fruit of its best checkpoint. That is similar in size to the dips on the original training curve, such as 0.33 between 4,096 and 5,120 steps. These are repeated looks at the same 30 episodes per seed, and no checkpoint was selected.
- **Episode sample.** The same default-setting DQN models scored 7.680 on the fresh evaluation seeds against 8.047 on the curve's seeds. The difference of about 0.4 fruit between two 150-episode samples is a useful scale for reading small gaps.

**Limits.** This is one small synthetic game. The held-out settings change two parameters that the agent can observe directly; they add no new mechanics, visuals or engine. They test robustness to parameter shift within Slicer, not generalization to other games. Each group has five seeds. The in-setting references use the hyperparameters and budget written for the default setting, so on bomb-heavy settings, where positive rewards are rarer, they may simply need more steps; the "gain" readings are conditional on that budget. The primary metric ignores bomb hits by design, which is why bomb hits are reported beside it. SB3 references and tabular Q were not evaluated in this run; the SB3 references were evaluated on the same settings on 23 October under a separate protocol. The GridQuest bug-hunter protocol and its evidence were not changed.

## SB3 references on the curve and held-out settings, 23 October 2026

**Protocol (committed first).** `make sb3-protocol` wrote [`evidence/sb3_reference_protocol.json`](evidence/sb3_reference_protocol.json) from the `PROTOCOL` constant in `analysis/sb3_reference.py`. It was committed on its own (`c08e75c`) before any SB3 model was checkpointed, trained for 20,480 steps or evaluated on a held-out setting. Before that commit the only SB3 runs were unit tests: 1,024-step runs on a test-only setting (bomb share 0.1) that is not in the protocol. The file also stores the SHA-256 of the five earlier evidence and protocol files it compares against, and the run refuses to start if any of them changed. The protocol fixes:
- **Agents and training.** SB3 DQN and SB3 PPO with the hyperparameters of the 5,120-step reference runs, training seeds 0–4 (the code asserts five distinct seeds), 20,480 steps. SB3 defines DQN's exploration as a fraction of the total budget, so the fraction is set to 3,584 / 20,480 = 0.175. ε therefore falls from 1.0 to 0.05 over the same 3,584 absolute steps as the scratch curve; a test checks that the schedule reaches 0.05 at step 3,584 and not before. PPO has no budget-dependent schedule. Nothing was tuned.
- **Checkpoints.** A Stable-Baselines3 callback evaluates the model at the start of each rollout that falls on a multiple of 1,024 steps, which is after the gradient updates for every earlier step, and once more after training ends. That gives 21 checkpoints of one evolving model. Evaluation is greedy. A test shows that, for both agents, the final weights are identical with and without the callback.
- **Evaluation.** Curve: the scratch curve's 30 episodes (evaluation seed 1000 + training seed). Held-out: the eight settings of `slicer-difficulty-v1`, copied unchanged, on the scratch difficulty run's episodes (7000 + training seed), with in-setting references trained on each held-out setting. The random and scripted anchors are recomputed and must equal the committed ones.
- **Metrics and readings.** The curve threshold (mean fruit 5.0, misses censored) and the transfer-gap rule are those of the earlier protocols. A new descriptive *implementation gap* is SB3 minus its scratch twin at 20,480 steps, with the same bootstrap over two five-seed groups, read as "SB3 higher", "scratch higher" or "no measurable difference". Episode return is reported beside fruit score, descriptively.
- **Predictions.** The 5,120-step checkpoint should equal the old 5,120-step SB3 endpoint for both agents, seed for seed. At 10,240 steps, PPO should equal the old endpoint and DQN should not, because that run decayed ε over 7,168 steps.

`make sb3` then ran once, with four worker processes, in 2 minutes 59 seconds. It trained 80 models: 2 agents × 5 seeds on the default setting, plus 2 × 5 on each of the 7 held-out settings. The output is committed as [`evidence/sb3_reference_20480.json`](evidence/sb3_reference_20480.json), a compact copy of `runs/sb3_reference.json`. No figure was drawn for this run.

**Checks.** All three predictions held. SB3 DQN's and SB3 PPO's 5,120-step checkpoints equal the 18 October endpoint runs seed for seed (five-seed means 3.960 and 1.627), and SB3 PPO's 10,240-step checkpoint equals the 10,240-step run (1.767). SB3 DQN's 10,240-step checkpoint differs (6.700 here against 6.173), as predicted. The recomputed anchors equal the committed ones. A second `make sb3` reproduced every row and anchor (`matches_committed_evidence: true`).

| Agent | Step 0 | 5,120 | 10,240 | 20,480 (95% seed bootstrap) | Bomb hits at 20,480 | Seeds reaching 5.0 |
|---|---:|---:|---:|---:|---:|---:|
| SB3 DQN | 0.773 | 3.960 | 6.700 | 7.787 [6.873, 8.627] | 0.547 | 5/5; first at 5,120–10,240 steps (median 8,192) |
| SB3 PPO | 1.153 | 1.627 | 1.767 | 2.413 [1.653, 3.227] | 0.167 | 0/5; all censored at 20,480 |
| Scratch Double DQN (20 October) | 0.773 | 3.513 | 5.920 | 8.047 [7.367, 8.727] | 0.513 | 5/5 |
| Scratch PPO (20 October) | 0.293 | 1.800 | 1.460 | 2.367 [1.353, 3.913] | 0.187 | 1/5 |

- **The library and scratch agents learned about the same.** At 20,480 steps on the curve episodes the implementation gap was −0.260 [−1.460, 0.793] for DQN and 0.047 [−1.607, 1.407] for PPO. Both read "no measurable difference", and so did all 16 agent-and-setting gaps on the held-out episodes. With five seeds this does not show the implementations are equivalent. Both DQNs crossed 5.0 at a median of 8,192 steps. SB3 DQN's curve flattened later in training: it stayed between 6.98 and 8.11 from 11,264 steps on. SB3 PPO rose slowly, from 0.600 at 1,024 steps to 2.413, and no seed reached 5.0.
- **Identical step-0 DQN scores.** SB3 DQN's untrained scores equal the scratch DQN's seed for seed. A check after the run found the reason: for seed 0, SB3's untrained Q-network has exactly the same initial weights as the scratch `QNetwork`, since both build 6→64→64→4 ReLU layers in the same order from the same Torch seed. Training then diverges.

**Held-out settings.** Each cell is a five-seed mean of 30-episode means at 20,480 steps, with the 95% seed bootstrap interval. The anchors are the same as in the 22 October tables.

**SB3 DQN**

| Setting | Trained on default: fruit | Bomb hits | Normalized | Trained on this setting: fruit | Bomb hits | Gap and reading |
|---|---:|---:|---:|---:|---:|---|
| train | 7.487 [6.733, 8.247] | 0.487 | 1.06 | (same models) | | |
| no_bombs | 13.560 [12.987, 14.087] | 0.000 | 0.94 | 14.107 [13.813, 14.293] | 0.000 | 0.547 [−0.027, 1.154], no measurable difference |
| slow_0.7 | 5.720 [5.200, 6.273] | 0.393 | 1.10 | 5.960 [5.447, 6.473] | 0.540 | 0.240 [−0.513, 0.973], no measurable difference |
| bombs_0.4 | 4.353 [3.693, 5.033] | 0.887 | 1.19 | 3.740 [3.486, 3.940] | 0.140 | −0.613 [−1.413, 0.080], no measurable difference |
| bombs_0.6 | 2.247 [1.857, 2.627] | 0.967 | 1.26 | 1.880 [1.667, 2.020] | 0.007 | −0.367 [−0.827, 0.100], no measurable difference |
| fast_1.3 | 9.300 [8.440, 10.240] | 0.587 | 1.06 | 9.787 [8.853, 10.787] | 0.773 | 0.487 [−0.874, 1.707], no measurable difference |
| fast_1.6 | 10.747 [9.960, 11.827] | 0.713 | 1.06 | 10.020 [9.393, 10.800] | 0.060 | −0.727 [−1.980, 0.367], no measurable difference |
| bombs_0.4_fast_1.3 | 5.347 [4.467, 6.167] | 1.033 | 1.14 | 4.653 [4.133, 5.093] | 0.000 | −0.693 [−1.720, 0.213], no measurable difference |

**SB3 PPO**

| Setting | Trained on default: fruit | Bomb hits | Normalized | Trained on this setting: fruit | Bomb hits | Gap and reading |
|---|---:|---:|---:|---:|---:|---|
| train | 2.300 [1.727, 2.893] | 0.160 | 0.15 | (same models) | | |
| no_bombs | 3.413 [2.365, 4.600] | 0.000 | 0.12 | 4.227 [2.527, 6.040] | 0.000 | 0.813 [−1.187, 2.974], no measurable difference |
| slow_0.7 | 1.627 [1.227, 2.040] | 0.133 | 0.10 | 1.513 [1.047, 1.887] | 0.207 | −0.113 [−0.740, 0.473], no measurable difference |
| bombs_0.4 | 1.480 [1.133, 1.880] | 0.313 | 0.15 | 0.700 [0.387, 1.013] | 0.000 | −0.780 [−1.293, −0.273], transfer gain |
| bombs_0.6 | 0.820 [0.633, 1.013] | 0.420 | 0.06 | 0.360 [0.227, 0.480] | 0.000 | −0.460 [−0.693, −0.220], transfer gain |
| fast_1.3 | 2.887 [2.247, 3.540] | 0.193 | 0.18 | 2.867 [1.920, 3.813] | 0.407 | −0.020 [−1.133, 1.267], no measurable difference |
| fast_1.6 | 3.380 [2.633, 4.160] | 0.240 | 0.18 | 2.733 [2.533, 2.973] | 0.813 | −0.647 [−1.427, 0.160], no measurable difference |
| bombs_0.4_fast_1.3 | 1.880 [1.487, 2.307] | 0.380 | 0.21 | 0.887 [0.767, 1.027] | 0.000 | −0.993 [−1.440, −0.587], transfer gain |

- **No transfer loss, again.** No gap interval among the 14 SB3 pairs lies above zero. SB3 DQN read "no measurable difference" on all 7 settings. SB3 PPO read the same on 4 and "transfer gain" on the 3 bomb-heavy settings.
- **The bomb-heavy pattern repeats, more weakly for DQN.** SB3 DQN's gaps on the three bomb-heavy settings point the same way as the scratch DQN's "transfer gains" (−0.613, −0.367, −0.693), but every interval crosses zero. The default-trained SB3 DQN hit 0.89–1.03 bombs per episode there, against 0.00–0.14 for the references. Its mean episode return, which counts bombs and is reported descriptively, was lower than the references' on all three: 3.092 against 3.215, 0.895 against 1.482, and 3.943 against 4.273.
- **SB3 PPO's gains come from references that barely learned.** On the bomb-heavy settings the references scored 0.36–0.89 fruit and hit no bombs, and their returns were below the default-trained models' on all three (for example 0.186 against 0.688 on `bombs_0.4`). SB3 PPO's normalized score stayed between 0.06 and 0.21. On the training setting its interval, [0.040, 0.262], lies just above zero, whereas scratch PPO's included zero. It is still far below the scripted rule.

**Limits.** This is the same small synthetic game, the same eight settings and five seeds per group. The SB3 and scratch agents share nominal hyperparameters but differ in network layout (SB3 PPO uses separate policy and value networks), time-limit handling (SB3 bootstraps at truncation; the scratch trainers do not) and optimizer details, so the implementation gap is a reference check, not a controlled comparison. The in-setting references use the budget written for the default setting. As of 23 October, tabular Q and held-out GridQuest layouts for learning agents were unevaluated; the layouts were measured on 6 November with a new transferable view, and tabular Q still cannot transfer by construction.

## A new defect type: coin taken but not scored, 23 October 2026

**Protocol (committed first).** `make defect-protocol` wrote [`evidence/new_defect_protocol.json`](evidence/new_defect_protocol.json) from `playtest/new_defect.py`. It was committed on its own (`e9eadb4`) before any hunt on the new defect; before that, the defect had only been driven through single transitions in unit tests. The file stores the SHA-256 of the frozen held-out map protocol, its results and the original-map contract run. The run refuses if any of them changed or if the regenerated held-out maps differ. The protocol fixes:
- **Defect.** `coin_not_scored`: entering the coin tile for the first time in an episode sets the coin flag but never adds the coin's 1 point. Every later score stays inside the declared 0–11 range. Each earlier defect moves the player illegally, blocks a legal move or pushes the score out of range; this one does none of those. It is kept out of the four-defect list, so `make playtest` and `make unseen` are unchanged.
- **Oracles.** v1 is the frozen 18 October contract, unchanged: bounds, the score range and movement legality. v2 is new (`check_transition` in `playtest/oracle.py`). From the map's declared size, walls, door, key, coin and goal, and the declared scoring (coin 1, goal 10), it predicts the position, key, coin and score after each move and compares them with what the game reports. Neither reads `info["bug_detected"]`. v2 was written in the same session as the defect, so it is not a blind test; v1 is.
- **Maps, search and scoring.** The default map and the eight frozen held-out maps of `gridquest-unseen-v1`, read from their protocol file, which is not modified. The coin is already in a different place on each map. The four policies, hunt seeds 0–4 and 1,200 steps per hunt. A hunt detects the defect at the first step either oracle reports any violation on the defective map; misses are right-censored at 1,200 steps. Detections out of 5 (default) and 40 (held-out) with Wilson intervals, mean and median step among detections only, and false alarms on a clean copy of every map for both oracles.
- **Prediction.** Written before the run: v1 detects nothing, because the missing point leaves the score in range; v2 flags `score_mismatch` at the first coin entry. Any v2 false alarm would be an oracle failure.

`make defect` then ran once, in about 4 seconds: 720 hunts, half on defective maps and half on clean copies. The output is committed as [`evidence/new_defect_results.json`](evidence/new_defect_results.json); a second `make defect` reproduced every row (`matches_committed_evidence: true`).

| Policy | v1, default map | v1, held-out maps | v2, default map | v2, held-out maps |
|---|---:|---:|---:|---:|
| Random | 0/5 | 0/40 | 5/5, 46.2 steps | 40/40, 117.6 |
| Scripted | 0/5 | 0/40 | 0/5 | 5/40, 2.0 |
| Coverage heuristic | 0/5 | 0/40 | 5/5, 36.6 | 40/40, 41.1 |
| Count-bonus Q | 0/5 | 0/40 | 5/5, 296.6 | **21/40**, 381.9 |

Each cell gives detections and the mean step among detections only. The Wilson intervals, medians, per-map counts and coverage are in the evidence file.
- **The frozen contract is blind to this defect.** v1 detected it in 0 of 180 defective hunts (4 policies × 45), exactly as predicted. A score that is wrong but plausible passes a range check.
- **The full-state contract caught it with no false alarms.** Every v2 detection was labelled `score_mismatch`. Neither oracle raised anything on the 360 clean hunts (0/5 and 0/40 for every policy and oracle). The v1 clean hunts on the held-out maps reproduced all 160 clean rows of the frozen 20 October results, which shows that adding the oracle option left v1 hunts unchanged.
- **Detection time is the time to reach the coin.** Random and coverage found it on every map and seed (Wilson [0.912, 1.0] on the held-out maps). The scripted key-then-goal bot found it only where the coin lies on its route: 5 of 40, all on one map, at step 2. Count-bonus Q found it in 5 of 5 on the default map but 21 of 40 held-out (Wilson [0.375, 0.671]) on 7 of 8 maps, which repeats its map sensitivity from 20 October.

**Limits.** This adds one defect type to one small authored game. v2 is a complete reference model of GridQuest's public state, which is possible only because the game's rules fit in a few lines; a real engine would not offer such a specification. v2 was designed knowing the defect, so its success here is not evidence that it would catch defects nobody anticipated. It was not run on the four earlier defects under a protocol; a unit test only checks that one coverage hunt flags each of them. The policies were not changed or tuned.

### Earlier checks after the 23 October runs

After both runs, `make test` passed 82 tests: 51 earlier ones, 15 in `tests/test_sb3_reference.py` and 16 in `tests/test_new_defect.py`. `make curve` reported `matches_protocol: true`, with rows identical to `evidence/learning_curve_20480.json`. `.venv/bin/python -m analysis.curves` gave the same four checks as before, and its consistency file was unchanged. `make unseen` reproduced the held-out bug-hunt rows and summary exactly; the regenerated file differed only in its timestamps and was restored. `make difficulty` printed `curve_consistency_identical: true` and `matches_committed_evidence: true`. All runs used this laptop (macOS, 11 CPU cores, CPU only) with at most four worker processes.

## Frozen-oracle reward challenge, 31 October 2026

The [reward-fault protocol](evidence/frozen_oracle_reward_protocol.json) was committed as `c0fd154` before `goal_reward_missing` was implemented or hunted. It fixes the unchanged v2 oracle, four policies, five distinct seeds, the default and eight frozen held-out maps, and a 1,200-step cap. Its SHA-256 entries freeze the oracle, the pre-extension hunt code, the map protocol and the prior coin-defect evidence. The only hunt-code change was adding the new defect to its allow-list; the runner checks that removing that one addition reproduces the frozen hash. Neither `check_transition` nor the search policy or stopping rule changed.

On successful key-and-goal entry, the injected environment returns **0.0** reward where the clean sparse game returns **1.0**. It leaves position, key, coin, score, termination and reported success unchanged. The v2 oracle compares only position, key, coin and score, so the protocol predicted a miss. A separate scripted key-then-goal probe, outside the hunt and oracle, observed the faulty reward on **45/45** map/seed runs. This establishes exposure for the scripted path; exposure of each random, coverage or count-bonus hunt was not measured.

The [saved run](evidence/frozen_oracle_reward_results.json) contains **360 hunts**: 180 defective and 180 clean. Every policy had **0/5** detections on the default map and **0/40** on the held-out maps, with every miss right-censored at 1,200 steps. The Wilson 95% upper bound is 0.4345 for each five-run cell and 0.0876 for each 40-run cell. The oracle raised **0/180** false alarms on clean hunts. No discovery-time mean or median exists because there were no detections. A repeat run matched every committed hunt and exposure row. The old `make defect` still matched its committed evidence, and `make unseen` reproduced its counts and zero clean false alarms; its timestamp-only evidence change was restored. The local suite passed **84 tests** after this addition.

This is a deliberately adversarial negative result, chosen after reading the v2 oracle's field list. It proves that this GridQuest contract cannot catch a reward-only fault even when a scripted path exposes it. It does not estimate the frequency of such faults in other games, the performance of a reward-aware oracle, or discovery quality for policies whose exposure was not measured. The earlier v1/v2 coin-defect result remains separate. The later reward-aware oracle has its own protocol and result in the [6 November section](#reward-aware-oracle-6-november-2026); v2 itself did not change.

## Longer bomb-heavy scratch DQN references, 31 October 2026

The [protocol](evidence/slicer_bomb_dqn_40960_protocol.json) was committed alone as `44a96ec` before the longer runs. It fixes the three bomb-heavy Slicer settings, scratch Double DQN, seeds 0–4, 20,480 and 40,960 steps, absolute ε decay over 3,584 steps, the same 30 greedy evaluation episodes per seed, and bomb hits and episode return as primary metrics. The runner hashes the frozen agent, environment, prior runner and evidence, and refuses a changed protocol. All **15/15** 20,480-step checkpoints reproduced the prior in-setting reference exactly in parameters, fruit, bomb hits and return. The [saved run](evidence/slicer_bomb_dqn_40960_results.json) has the seed rows; a second run matched them.

| Setting | Step | Fruit score (95% seed interval) | Bomb hits/episode (95% seed interval) | Episode return (95% seed interval) |
|---|---:|---:|---:|---:|
| `bombs_0.4` | 20,480 | 3.653 [3.267, 4.053] | 0.000 [0.000, 0.000] | 3.270 [2.882, 3.673] |
| `bombs_0.4` | 40,960 | 3.760 [3.407, 4.133] | 0.000 [0.000, 0.000] | 3.377 [3.022, 3.754] |
| `bombs_0.6` | 20,480 | 1.613 [1.306, 1.833] | 0.000 [0.000, 0.000] | 1.212 [0.898, 1.441] |
| `bombs_0.6` | 40,960 | 1.833 [1.587, 2.033] | 0.027 [0.000, 0.080] | 1.413 [1.191, 1.590] |
| `bombs_0.4_fast_1.3` | 20,480 | 4.607 [4.167, 5.027] | 0.033 [0.000, 0.100] | 4.193 [3.739, 4.647] |
| `bombs_0.4_fast_1.3` | 40,960 | 4.180 [3.240, 4.867] | 0.053 [0.000, 0.160] | 3.726 [2.794, 4.434] |

The cautious bomb-hit pattern persisted at the doubled budget for these settings and seeds, while fruit and return moved in different directions. The two checkpoints reuse evaluation episodes and come from the same evolving models, so this table is descriptive, without a paired significance claim. At this **31 October** checkpoint, the default-trained agents were **not** extended; comparing a 40,960-step in-setting model with a 20,480-step transferred model would have confounded training budget. The next section closes that specific gap on the three known settings. Other settings and new seeds remain unmeasured.

## Budget-matched bomb-heavy comparison, 1 November 2026

The [protocol](evidence/slicer_bomb_matched_40960_protocol.json) was committed alone as `e35f7de` before any extended default-trained or PPO/SB3 model ran. It freezes the four agent types, three previously inspected bomb-heavy settings, seeds 0–4, 40,960-step budget, 20,480-step reproduction checkpoint, fixed 3,584-step DQN ε decay and 30 evaluation episodes per seed. The run trained 65 models and reused the 15 committed longer scratch DQN in-setting rows. All **80/80** 20,480-step checkpoints matched the earlier parameters and fruit, bomb-hit and return values. A second run matched all saved rows and summaries in the [evidence](evidence/slicer_bomb_matched_40960_results.json).

At 40,960 steps, each cell below compares a default-trained model with one trained directly on that setting. Values are five-seed means; gap is **in-setting minus default-trained**, with a 95% interval from the predeclared 2,000-resample independent-group bootstrap. Complete seed values and their own intervals are in the evidence.

| Agent | Setting | Fruit default / in-setting | Bomb hits default / in-setting | Return default / in-setting | Fruit gap [95% CI] | Return gap [95% CI] |
|---|---|---:|---:|---:|---:|---:|
| Scratch DQN | `bombs_0.4` | 5.193 / 3.760 | 1.787 / 0.000 | 3.042 / 3.377 | -1.433 [-1.887, -0.960] | 0.335 [-0.213, 0.913] |
| Scratch DQN | `bombs_0.6` | 2.840 / 1.833 | 2.240 / 0.027 | 0.225 / 1.413 | -1.007 [-1.333, -0.693] | 1.187 [0.649, 1.744] |
| Scratch DQN | `bombs_0.4_fast_1.3` | 6.293 / 4.180 | 2.133 / 0.053 | 3.802 / 3.726 | -2.113 [-3.073, -1.340] | -0.076 [-1.105, 0.781] |
| Scratch PPO | `bombs_0.4` | 2.287 / 1.900 | 0.000 / 0.000 | 1.851 / 1.449 | -0.387 [-1.147, 0.393] | -0.402 [-1.196, 0.405] |
| Scratch PPO | `bombs_0.6` | 1.247 / 0.860 | 0.000 / 0.000 | 0.826 / 0.419 | -0.387 [-0.753, -0.020] | -0.407 [-0.798, -0.013] |
| Scratch PPO | `bombs_0.4_fast_1.3` | 2.880 / 1.720 | 0.000 / 0.000 | 2.428 / 1.217 | -1.160 [-2.180, -0.260] | -1.211 [-2.286, -0.250] |
| SB3 DQN | `bombs_0.4` | 5.647 / 3.907 | 2.367 / 0.220 | 2.920 / 3.307 | -1.740 [-3.173, -0.587] | 0.387 [-0.173, 0.897] |
| SB3 DQN | `bombs_0.6` | 3.180 / 1.873 | 3.033 / 0.000 | -0.222 / 1.482 | -1.307 [-2.540, -0.413] | 1.704 [0.920, 2.664] |
| SB3 DQN | `bombs_0.4_fast_1.3` | 6.747 / 4.613 | 2.607 / 0.140 | 3.786 / 4.094 | -2.133 [-4.053, -0.727] | 0.308 [-0.268, 0.862] |
| SB3 PPO | `bombs_0.4` | 2.493 / 2.073 | 0.060 / 0.000 | 2.002 / 1.627 | -0.420 [-1.373, 0.600] | -0.376 [-1.384, 0.695] |
| SB3 PPO | `bombs_0.6` | 1.327 / 0.413 | 0.073 / 0.000 | 0.834 / -0.056 | -0.913 [-1.193, -0.580] | -0.891 [-1.225, -0.517] |
| SB3 PPO | `bombs_0.4_fast_1.3` | 3.053 / 2.347 | 0.080 / 0.000 | 2.526 / 1.862 | -0.707 [-2.360, 0.967] | -0.664 [-2.382, 1.066] |

For both DQNs, the default-trained fruit advantage persisted at matched budget in all three settings, but the default-trained models hit about 1.8–3.0 bombs per episode versus 0.0–0.2 for in-setting references. On `bombs_0.6`, the return gap favored in-setting training for both DQNs. PPO results did not follow one pattern: scratch PPO's default-trained model had higher fruit and return on two settings under the protocol's interval rule; SB3 PPO had that reading on `bombs_0.6` only. These twelve cells were all reported without selecting a winner. They reuse settings already examined on 22–23 October, five seeds and the same evaluation episodes; there is no fresh held-out test, multiple-comparison adjustment or claim of equivalence when an interval crosses zero. The earlier 20,480-step results keep their original scope.

## New-setting transfer check, 2 November 2026

The [protocol](evidence/slicer_fresh_transfer_40960_protocol.json) was committed alone as `04a4ab8` before training or scoring either new setting. It fixed four agent types, **seven** training seeds (0–6), two previously unused bomb-share/speed combinations, 40,960 training steps, 30 greedy evaluation episodes per seed from a new 9,100 seed block, the existing hyperparameters and random/scripted anchors. The runner was committed as `435c887` before the run. Its **84/84** jobs completed, and the default-trained models' parameter fingerprints at 20,480 and 40,960 steps reproduced **40/40** earlier checkpoints for seeds 0–4. A second full run matched every row, anchor and summary in the [saved first result](evidence/slicer_fresh_transfer_40960_results.json).

The protocol prespecified **SB3 DQN mean episode return on `stress_0.5_fast_1.45`** as the primary comparison. Its in-setting-minus-default gap was **+1.223 return per episode [0.466, 2.076]** under the seed-level, 2,000-resample bootstrap: a transfer loss by the declared interval rule. The default-trained model sliced **5.295** fruit versus **3.590** in-setting but hit **2.952** bombs versus **0.000**, explaining why fruit alone points the other way. The table reports all eight agent/setting return comparisons. Values are seven-seed means; each slash is default-trained / in-setting, and each gap is in-setting minus default-trained. Full per-seed values and group intervals are in the saved evidence.

| Agent | New setting | Fruit | Bomb hits | Return | Return gap [95% CI] | Reading |
|---|---|---:|---:|---:|---:|---|
| Scratch DQN | `intermediate_0.3_slow_0.85` | 5.467 / 4.705 | 1.390 / 0.614 | 3.709 / 3.715 | +0.006 [-0.435, +0.388] | No measurable difference |
| Scratch DQN | `stress_0.5_fast_1.45` | 5.205 / 3.529 | 2.762 / 0.000 | 2.081 / 3.141 | +1.060 [+0.458, +1.718] | Transfer loss |
| Scratch PPO | `intermediate_0.3_slow_0.85` | 2.629 / 2.081 | 0.048 / 0.000 | 2.149 / 1.639 | -0.510 [-1.249, +0.252] | No measurable difference |
| Scratch PPO | `stress_0.5_fast_1.45` | 2.395 / 1.052 | 0.138 / 0.000 | 1.817 / 0.531 | -1.285 [-1.749, -0.793] | Transfer gain |
| SB3 DQN | `intermediate_0.3_slow_0.85` | 5.762 / 4.562 | 1.562 / 0.576 | 3.835 / 3.609 | -0.226 [-0.599, +0.160] | No measurable difference |
| **SB3 DQN (primary)** | `stress_0.5_fast_1.45` | 5.295 / 3.590 | 2.952 / 0.000 | 1.983 / 3.206 | **+1.223 [+0.466, +2.076]** | **Transfer loss** |
| SB3 PPO | `intermediate_0.3_slow_0.85` | 2.776 / 2.319 | 0.038 / 0.000 | 2.312 / 1.871 | -0.441 [-1.128, +0.260] | No measurable difference |
| SB3 PPO | `stress_0.5_fast_1.45` | 2.481 / 1.143 | 0.076 / 0.000 | 1.966 / 0.624 | -1.342 [-2.068, -0.727] | Transfer gain |

The random/scripted anchor mean returns were **0.203 / 3.826** on the intermediate setting and **-0.582 / 3.323** on the stress setting. They give context for achievable returns under these authored policies, not an optimality bound. The intermediate setting gave no return-gap interval entirely on one side of zero for any agent. On the stress setting, both DQNs favored in-setting training on return and both PPOs favored default training, even as all four default-trained models sliced more fruit. The seven secondary return intervals and all fruit comparisons are descriptive and unadjusted for multiple testing; an interval crossing zero is not evidence of equivalent performance.

These combinations were selected by the project author after seeing prior bomb-heavy results, then frozen before their own training and evaluation. They were absent from prior setting lists, so this is a new within-game setting check, **not** an independently designed challenge, real-game transfer estimate or proof that the named training method is generally superior. Episodes share seeded game rules, and the 30 evaluations of one trained model are not 30 independent training replicates. No agent, hyperparameter, reward or comparison rule was changed after seeing these scores.

## Reward-aware oracle, 6 November 2026

**Question.** Does a new contract, v3, report the known reward fault without new false alarms and without losing any detection the frozen v2 contract makes? This is a regression check on a known blind spot. **The same author wrote v3 knowing the reward fault, so detecting it is not an independent discovery.**

**The oracle.** `check_transition_v3` in `playtest/reward_oracle.py` calls the unchanged v2 `check_transition` first and returns any state violation. If the state is right, it compares the returned reward with the declared table and reports `reward_mismatch` outside an absolute tolerance of 1e-9. In sparse mode, the hunted mode, the table pays 1.0 on entering the goal with the key and -0.01 otherwise, and adds -0.05 when the resulting cell is a hazard. The oracle predicts the reward from the expected state and reads neither `info["success"]` nor `info["bug_detected"]`. `playtest/oracle.py`, `playtest/bug_hunt.py` and `envs/gridquest.py` are byte-identical to the 2 November commit, because earlier protocols pin them; v3 therefore has its own copy of the hunt loop.

**Protocol first.** [`evidence/reward_oracle_protocol.json`](evidence/reward_oracle_protocol.json) (SHA-256 `7e447803…`) was committed alone as `95dfcc2` at 09:14:37 UTC, and the scored run started at 09:15:57 UTC from `90da015`. It pins v3's code (committed earlier as `02541d4`), the frozen oracle, hunt, game, map generator and earlier evidence. The project's implementing agent chose every setting, reusing the maps, policies, seeds and budget of the two earlier defect protocols, after reading their results, including v2's 0/180 and the 45/45 scripted exposure. Before the protocol, v3 had run only on hand-written transitions and clean default-map random play at seeds 1000–1003. After the protocol and before the scored run, the runner's unit tests also ran one scripted default-map fault hunt at seed 0 and four clean 600-step hunts at seed 7.

**Design.** Seven conditions (a clean copy, the five earlier defects, the reward fault) × two oracles × four policies × 45 map/seed pairs (the default map and eight frozen held-out maps, five seeds each), at most 1,200 steps: **2,520 hunts**, 180 per oracle and condition. Misses are right-censored. Each v3 row also records the first step reporting success, which measures whether the hunter reached the goal with the key and never feeds the oracle.

**Acceptance: PASS on all four predeclared rules, on the first scored run.** No setting was changed and no run was repeated to obtain this.

| Rule | Result |
|---|---|
| A. No v3 false alarms | 0/180 clean hunts (Wilson 95% upper bound 0.0209) |
| B. Reward fault | Scripted 45/45. All 130 exposed hunts were detected as `reward_mismatch` at the first exposure; none of the 50 unexposed hunts was detected |
| C. No regression against v2 | 900 paired hunts on the five earlier defects: none lost or later, and no difference in step, kind or coverage |
| D. Controls | v2 still 0/180 on the reward fault. Its 180 reward-fault, 180 coin and 180 clean rows equal both earlier evidence files. v2 and v3 clean hunts had identical visit counts on 180/180 |

**Reward fault, detections per policy** (mean step among detections only; misses censored at 1,200):

| Policy | v2 default | v2 held-out | v3 default [Wilson 95%] | v3 held-out [Wilson 95%] | Censored under v3 |
|---|---:|---:|---:|---:|---:|
| Random | 0/5 | 0/40 | 4/5, step 570.0 [0.376, 0.964] | 36/40, step 314.3 [0.770, 0.960] | 5 |
| Scripted | 0/5 | 0/40 | 5/5, step 10.0 [0.566, 1.000] | 40/40, step 8.4 [0.912, 1.000] | 0 |
| Coverage | 0/5 | 0/40 | 5/5, step 126.2 [0.566, 1.000] | 39/40, step 82.7 [0.871, 0.996] | 1 |
| Count-bonus Q | 0/5 | 0/40 | 0/5 [0.000, 0.434] | 1/40, step 1,124.0 [0.004, 0.129] | 44 |

Overall, v3 detected **130/180** [0.653, 0.783], against v2's **0/180** [0.000, 0.021]. **The 50 misses are not oracle silences:** none of those hunts reached the goal with the key inside 1,200 steps, and 44 of them belong to the count-bonus policy, which rarely reaches the goal. Reaching the fault limits detection here, as with any oracle that only sees where a hunter goes.

**Earlier defects, out of 180 hunts per oracle** (v2 and v3 rows are identical; this is also the first protocol run of v2 on the four original defects):

| Defect | v2 and v3 detected | Wilson 95% | Violation kind |
|---|---:|---:|---|
| `wall_gap` | 100/180 | [0.483, 0.626] | `position_mismatch` |
| `score_overflow` | 107/180 | [0.522, 0.664] | `score_mismatch` |
| `unreachable_checkpoint` | 160/180 | [0.835, 0.927] | `position_mismatch` |
| `softlock` | 125/180 | [0.624, 0.757] | `position_mismatch` |
| `coin_not_scored` | 121/180 | [0.601, 0.737] | `score_mismatch` |

The unchanged misses on these defects are censored as well; v3 adds nothing to them, because none of these faults changes the reward. Every row is in [`evidence/reward_oracle_results.json`](evidence/reward_oracle_results.json).

**Reproduction.** A second full execution (started 09:16:31 UTC, from `04e6cfa`) matched the saved rows, summary, acceptance and controls exactly. After it, `make reward-challenge` and `make defect` still matched their evidence, and `make unseen` changed only its evidence file's timestamps, which were restored. Training evidence was not rerun, because no agent, game, analysis or experiment file changed. The suite passed **113 tests** (86 before).

**Limits.**

- **Not an independent discovery.** The fault was known to the author and the maps and seeds were already inspected. Independent challenge design remains open.
- **The reward table is not an outside specification.** It was transcribed from the clean reward branch of `GridQuest.step`; the earlier docs state the 1.0 goal payment and a per-step cost in words, while -0.01 and -0.05 appeared only in that code. A mistake in the clean rule would be copied into the table and go unnoticed.
- **One reward fault was hunted.** Other wrong amounts (2.0 at the goal, a wrong step cost, a missing hazard penalty) are covered only by unit tests on hand-written transitions.
- **Only sparse mode was hunted.** The shaped, fixed and hackable tables have unit tests and clean random play on the default map, nothing more.
- **v3 does not check** `terminated`, `truncated`, `info["success"]` or the observation integer.
- **Scope.** Policy differences are descriptive, with no significance test. This is one small authored game, and it says nothing about how often reward faults occur or how an oracle would behave on an external engine.

## Held-out GridQuest layouts for a learning agent, 6 November 2026

**Question.** Can a PPO agent trained on seeded GridQuest layouts play layouts it never saw? Until this run every GridQuest learner read the game's single integer observation, which carries position and two flags and nothing about the layout, so a learner could only memorise one map. Held-out GridQuest layouts for a learning agent were listed as open from 20 October onward; this section measures them.

**What the code adds.** `envs/gridquest_views.py` puts two array views beside the unchanged integer observation: **egocentric**, seven 15×15 planes centred on the player (blocked including the off-map area, door, untaken key, goal, hazard, untaken coin, has-key), which covers the whole map at sizes 6, 7 and 8 and shows no defect-only cell; and **absolute**, the integer observation's own information as one-hot columns, kept as a negative control that cannot generalise by construction. `LayoutPool` draws one layout per episode. `agents/layout_ppo.py` is a scratch clipped PPO that reuses `gae_advantages` from `agents/ppo.py`, with a conv 16 (3×3, stride 2) / conv 32 (3×3) / linear 128 body for the egocentric view and two 128-unit linear layers for the control. `playtest/transfer_layouts.py` draws layouts from the frozen `playtest/levels.py` generator and validator and rejects, in draw order and before any agent plays, anything invalid, unsolvable, or an exact or near duplicate of an accepted or reserved layout. GridQuest itself, its integer observation, the oracles, the bug hunts and every earlier protocol and evidence file are unchanged.

**Protocol first.** [`evidence/gridquest_transfer_protocol.json`](evidence/gridquest_transfer_protocol.json) (SHA-256 `4f835e90…`) was committed alone as `f33050b` before any training or scoring on these layouts, and the guarded runner, `make gridquest-transfer` and its 17 tests followed as `e475cfd`, still before the first scored run. The protocol fixes the generator and its three set seeds, the SHA-256 of all three layout sets, the complete disjointness report, seven frozen input hashes, the two views, both networks, the shaped reward used for training, a 1,024,000-step budget per run, the four-setting validation search, three search seeds, seven final seeds, the primary metric, the bootstrap and three acceptance rules. The runner refuses to start if any of that differs, and the project's implementing agent chose every setting after seeing its own 15 disclosed smoke runs on reserved maps — nobody independent reviewed them.

**Layouts.** 10,000 training, 100 validation and 200 test layouts from three separate generator seeds, cycling sizes 6, 7 and 8, every one solvable. Two same-size layouts are near duplicates when their labelled cells (wall, hazard, start, key, coin, door, goal) differ in at most 4 cells under any of the 8 rotations and reflections of the square. The default maps, the eight earlier bug-hunt maps and the 400 disclosed smoke maps are registered first, so they can never enter a set. The run regenerated every set, matched all three set hashes, and recomputed the disjointness report: no shared content hash in any of the 12 set pairs, minimum cell distance 5 to 11, `all_disjoint: true`. Training is cut into blocks of 1,000 maps, one per seed slot, and a condition with *N* layouts takes the first *N* of its own block, so no training map is shared between seeds. The test layouts were first handed to a job only after validation selection was fixed.

**Selection, on validation only.** Four learning-rate/entropy settings trained at 1,000 layouts with the three search seeds and scored on the 100 validation layouts: `lr3e-4_ent0.03` **0.9109** (chosen), `lr1e-3_ent0.01` 0.9072, `lr3e-4_ent0.01` 0.9044, `lr1e-3_ent0.03` 0.8939. That one setting then ran every condition, so the 1-, 10- and 100-layout runs and the absolute control use a setting chosen for 1,000 layouts. The protocol discloses this rather than correcting it.

**The run.** The first scored execution ran once with 8 worker processes, from 15:11:26 to 15:40:31 UTC on 6 November (23:11–23:40 Singapore time), 29 minutes 5 seconds, from code commit `e475cfd`: 12 search jobs and 42 final jobs (6 conditions × 7 seeds), 54 rows, each consuming the full 1,024,000 steps. Its output is committed unchanged as [`evidence/gridquest_transfer_results.json`](evidence/gridquest_transfer_results.json), saved on 7 November before any of this wording was written.

**Acceptance: PASS on all three predeclared rules, on the first scored run.** No setting was changed and no scored run was repeated to obtain it.

| Rule | Contrast on the 200 test layouts, sampled SPL | Result |
|---|---|---|
| A. Beats random by the declared margin | `egocentric_1000` − random = **+0.8547 [0.8304, 0.8771]**; rule asks for a lower bound of at least 0.50 | PASS |
| B. Beats the single-layout reference | `egocentric_1000` − `egocentric_1` = **+0.8859 [0.8591, 0.9096]**; rule asks for a lower bound above 0 | PASS |
| C. Beats the absolute control | `egocentric_1000` − `absolute_1000` = **+0.8309 [0.8057, 0.8538]**; rule asks for a lower bound above 0 | PASS |

**The protocol calls rule A a weak test, and it is.** The author set the 0.50 margin after smoke runs that already exceeded it, so a PASS there was expected. The informative numbers are the size of the score, the generalisation gap and the change with layout count.

Primary metric: SPL, `shortest / max(taken, shortest)` on success and 0 otherwise, with actions sampled from the policy, 5 episodes per layout, averaged within a layout then across layouts and seeds (7 final seeds). Intervals are 95% percentile bootstrap intervals crossed over layouts and seeds, 5,000 resamples.

| Condition | Own training layouts | Validation (100) | Test (200) | Test success | Test greedy SPL |
|---|---:|---:|---:|---:|---:|
| `egocentric_1000` | 0.9566 | 0.9084 | **0.9005 [0.8764, 0.9233]** | 0.9704 | 0.8759 |
| `egocentric_100` | 0.9793 | 0.7744 | 0.7751 [0.7369, 0.8127] | 0.8927 | 0.6745 |
| `egocentric_10` | 0.9713 | 0.3047 | 0.3152 [0.2634, 0.3726] | 0.5226 | 0.1988 |
| `egocentric_1` | 0.9832 | 0.0146 | 0.0146 [0.0079, 0.0236] | 0.0296 | 0.0079 |
| `absolute_1000` (control) | 0.0716 | 0.0726 | 0.0696 [0.0623, 0.0773] | 0.3971 | 0.0093 |
| `absolute_1` (control) | 0.9832 | 0.0075 | 0.0049 [0.0017, 0.0094] | 0.0110 | 0.0035 |
| Random policy | — | 0.0424 | 0.0457 | 0.3321 | — |
| Scripted search, knows the map | — | 1.0 | 1.0 | 1.0 (200/200, all on a shortest path) | — |

Layout count mattered: +0.3007 [0.2488, 0.3564] test SPL from 1 to 10 layouts, +0.4599 [0.3999, 0.5148] from 10 to 100, and +0.1254 [0.0975, 0.1532] from 100 to 1,000. The generalisation gap, a seed's own training layouts minus the test layouts, shrank the same way.

| Condition | Own training | Test | Gap [95%] |
|---|---:|---:|---:|
| `egocentric_1` | 0.9832 | 0.0146 | 0.9686 [0.9346, 0.9901] |
| `egocentric_10` | 0.9713 | 0.3152 | 0.6561 [0.5915, 0.7154] |
| `egocentric_100` | 0.9793 | 0.7751 | 0.2042 [0.1658, 0.2437] |
| `egocentric_1000` | 0.9566 | 0.9005 | **0.0561 [0.0328, 0.0820]** |
| `absolute_1` | 0.9832 | 0.0049 | 0.9783 [0.9420, 0.9981] |
| `absolute_1000` | 0.0716 | 0.0696 | 0.0020 [−0.0063, 0.0100] |

**Misses and failures, plainly.**
- **The gap never closed.** At 1,000 layouts it is 0.0561 with an interval entirely above zero, so a measurable amount of the score still comes from having trained on that particular map. It shrank from 0.2042 at 100 layouts; it did not vanish.
- **The agent is about 10% short of the map-knowing search.** The scripted reference scores 1.0 on all 200 test layouts; the lead agent scores 0.9005 and fails 2.96% of its test episodes outright within the game's own `size² × 3` step limit.
- **One test layout is not solved better than chance.** Of the 200, 199 beat the random mean and 1 did not; 191 reach SPL 0.5 or more; the worst layout scores 0.0072 and the best 1.0.
- **Greedy play is worse than sampled play everywhere**: 0.8759 against 0.9005 at 1,000 layouts and 0.6745 against 0.7751 at 100. A memoryless greedy policy can walk into a loop, which is why the protocol declared sampled actions as primary before the run.
- **The single-layout agent is worse than random off its own map**: 0.0146 test SPL against random's 0.0457, with 0.9832 on the one map it trained on. The protocol predicted exactly this.
- **The control was predicted to be "close to random" and came out somewhat above it**: `absolute_1000` scores 0.0696 against random's 0.0457 (success 0.3971 against 0.3321). Its own-training and test scores are the same to within 0.002, so it memorised nothing about layouts; with no view of the map it appears to have learned a weak layout-independent habit. The prediction was directionally right and not exact.
- **`egocentric_1` does not score 1.0 on its own map** (0.9832): one of the seven seeds scores 0.8824 there under sampled actions.
- **The chosen setting was selected at 1,000 layouts** and then applied to the 1-, 10- and 100-layout conditions and the control, which may understate them.

By test layout size the lead agent scored 0.8809 (size 6), 0.8938 (size 7) and 0.9271 (size 8), against random's 0.0531, 0.0434 and 0.0407. As a labelled secondary set, the eight earlier bug-hunt maps — seen by their developers, never by a learning agent, and read by no acceptance rule — gave 0.8824 [0.7301, 0.9912] for `egocentric_1000`, 0.7341 for `egocentric_100`, 0.3131 for `egocentric_10`, 0.0446 for `egocentric_1` and 0.0508 for `absolute_1000`.

**Checks.** `make test` passes **156 tests** (113 before): 26 added in `tests/test_gridquest_views.py` with the views, and 17 in `tests/test_gridquest_transfer.py` with the runner, before the first scored run. Apart from one new Makefile target, every file in this work is new, so no input of any earlier protocol changed and no earlier evidence was rerun. The run itself rechecks what it depends on at startup: the protocol hash, seven frozen input hashes, three layout-set hashes, the generation-rejection counts and the complete disjointness report.

**Limits declared in advance.** One small authored game. Train, validation and test all come from one same-author generator, so this measures interpolation inside that distribution, not transfer to another game or to a differently designed level set. The egocentric view hands the agent the complete map, which a real game may not. Seven seeds is few, so the bootstrap intervals are approximate and may be somewhat narrow. The run is clean play only: no defect is injected, and no oracle or bug hunt takes part.

## Open work

Checkpoint curves and held-out Slicer evaluation for both the scratch agents and the SB3 references, a fifth coin defect, a separate reward-fault challenge, a reward-aware oracle regression check, a matched-budget follow-up, a seven-seed new-setting check and held-out GridQuest layouts for an agent whose observation can transfer are measured above. Still open: independently designed Slicer settings, layout generators and defect samples, including a challenge for the reward-aware oracle that its author did not design and held-out layouts from a generator this project did not write; a bug hunt that uses the new egocentric view; an external-engine oracle, learned curiosity refinement, graphical pygame GIFs and an engine integration. The Python project remains headless; a separate browser-only port of clean GridQuest can be tried from the portfolio site. No game-engine deployment was made.
