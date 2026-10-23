# Measured results, 12–23 October 2026

All scores are from synthetic games created in this repository. Five independent training seeds per setting; intervals are percentile bootstrap intervals over **seed means** (2,000 resamples), not over individual episodes. Evaluation uses 30 Slicer or 100 GridQuest episodes per seed. These small games and training budgets do not imply performance in commercial titles. The exact seed rows and run configuration are committed in `evidence/`.

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

**Limits.** All eight maps come from one small game and one generator, and the defect types are the same four. This tests location generalization, not new kinds of bug or another engine (one new defect type was added on 23 October under its own protocol; see below). The learning agents (Q, DQN, PPO) were not evaluated on unseen maps. Tabular Q's observation is a map-specific position index and cannot transfer by construction. The neural agents were later evaluated on held-out Slicer settings instead (22 October section).

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

**Limits.** This is the same small synthetic game, the same eight settings and five seeds per group. The SB3 and scratch agents share nominal hyperparameters but differ in network layout (SB3 PPO uses separate policy and value networks), time-limit handling (SB3 bootstraps at truncation; the scratch trainers do not) and optimizer details, so the implementation gap is a reference check, not a controlled comparison. The in-setting references use the budget written for the default setting. Tabular Q and held-out GridQuest layouts for learning agents remain unevaluated.

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

## Open work

Checkpoint curves and held-out Slicer evaluation for both the scratch agents and the SB3 references, and one new defect type with a full-state contract, are measured above. Still open: more seeds, a longer budget for the bomb-heavy in-setting references, held-out GridQuest layouts for an agent whose observation can transfer, defect types nobody designed an oracle around, external-engine bug oracles, learned curiosity refinement, graphical pygame GIFs and an engine integration. A browser or game-engine deployment was not made.
