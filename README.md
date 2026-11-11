# PlaytestRL

PlaytestRL uses two small Gymnasium games to ask how an automated player learns and whether it can expose a broken game rule. GridQuest makes reward-design mistakes visible and now includes a held-out-layout learning study; Slicer supports bounded comparisons of hand-written and library agents under longer and newly combined difficulty settings. A separate playtest harness measures how four search policies find planted defects and tests a reward fault against a frozen oracle.

![Seeded agent endpoint comparison](figures/learning_comparison.png)

## Why this project

Manual game testing can miss paths that nobody thought to script. An agent may reach those paths, but it may also learn to farm a badly designed reward. The games and injected defects here are small enough to inspect, so both successes and failures have an explanation.

## Current milestones

| Area | What works today | Next question |
|---|---|---|
| Games and agents | Checked GridQuest/Slicer environments, tabular Q-learning, scratch Double DQN/PPO and library references. A predeclared 40,960-step check on two newly combined Slicer settings found the primary SB3 DQN in-setting return ahead of default-trained by +1.223 [0.466, 2.076] on the harder setting | These are same-author, synthetic game settings; the earlier seven-setting result remains a separate dated study |
| Reward and bug studies | The original oracle missed a coin-not-scored fault in 0 of 180 hunts. The later full-state v2 contract caught that known fault. A frozen v2 then missed a predeclared goal-reward fault in 0 of 180 defective hunts; the reward-aware v3 caught it in 130 of 180, with 0 of 180 clean false alarms | The v3 repair was written knowing the fault; an independent challenge remains open |
| Training evidence | Five-seed 20,480-step checkpoint curves for scratch and SB3 agents; longer bomb-heavy comparisons at 40,960 steps; a GridQuest PPO trained on generated layouts and tested on 200 held-out layouts reached SPL 0.9005 [0.8764, 0.9233] | Neither PPO converged on Slicer within 20,480 steps. GridQuest train, validation and test share one generator; its train-test gap remains 0.0561 [0.0328, 0.0820] |

## Run locally

You need Python 3.12.

```sh
python3.12 -m venv .venv
. .venv/bin/activate
pip install -r requirements.txt
make test
make all
```

`make all` runs the Q-learning, neural, playtest, ablation, and figure pipelines. [Results](RESULTS.md) reports the seeded outcomes and what each metric means.

A separate [10,240-step neural run](evidence/neural_10240.json) doubles the earlier interaction budget for the same four agent families and five seed numbers. It starts fresh models and retains every seed row. The DQN endpoints improved in this small game, but exploration schedules also scale with the chosen budget, so these two runs do not form a checkpoint learning curve. The exact scores and intervals are in [Results](RESULTS.md).

`make curve` trains one Double DQN and one PPO model per seed for 20,480 steps and evaluates the same evolving model every 1,024 steps; `make curve-figure` draws [the curve](figures/learning_curve.png) from the committed [evidence](evidence/learning_curve_20480.json). Its 5,120-step checkpoints reproduce the original endpoint run exactly.

`make difficulty` runs the Slicer difficulty protocol frozen in [a protocol file](evidence/slicer_difficulty_protocol.json) before any result existed. Scratch Double DQN and PPO are trained on the default game with the curve settings (five seeds, 20,480 steps) and evaluated unchanged on seven held-out bomb-share and speed settings, against fresh agents trained on each setting and random and scripted anchors. No agent showed a measurable transfer loss. DQN's apparent fruit-score gains on the bomb-heavy settings came with more bomb hits, and PPO was barely above random everywhere ([figure](figures/slicer_difficulty.png), [evidence](evidence/slicer_difficulty_20480.json)). The default-setting models reproduce the committed curve checkpoints 210 of 210.

`make sb3` applies the same protocol design to the Stable-Baselines3 DQN and PPO references under [their own frozen protocol](evidence/sb3_reference_protocol.json). A callback checkpoints each model every 1,024 steps up to 20,480. SB3 DQN reached the predeclared threshold in 5 of 5 seeds and SB3 PPO in none; every comparison with the scratch agents read "no measurable difference", and no SB3 agent showed a transfer loss on the held-out settings ([evidence](evidence/sb3_reference_20480.json)).

`make defect` hunts a fifth defect, `coin_not_scored`, under [a protocol frozen first](evidence/new_defect_protocol.json) on the default map and the eight held-out maps. The original contract never saw it (0/180). A full-state contract that predicts position, key, coin and score after every move caught it in every random and coverage hunt, 21 of 40 held-out count-bonus hunts and 5 of 40 scripted hunts, with no false alarm in 360 clean hunts. That contract was written knowing the defect, so this is a fix, not a blind test ([evidence](evidence/new_defect_results.json)).

`make reward-challenge` replays a protocol committed before the scored run against the unchanged v2 oracle. A successful goal returned zero instead of the declared reward one while public position, key, coin and score stayed unchanged. The oracle missed the fault in all 180 defective hunts, though a separate scripted probe exposed it on all 45 map/seed runs. `make reward-oracle` applies a reward-aware v3 contract under a separate protocol: it caught the known fault in 130 of 180 hunts, including all 45 scripted hunts. The remaining 50 never reached the relevant goal state. Neither result is a blind, externally designed defect challenge ([results](RESULTS.md)).

`make bomb-dqn-longer` and `make bomb-matched-longer` revisit three known bomb-heavy settings at 40,960 steps. `make fresh-transfer` uses a separately committed protocol on two previously unused Slicer setting combinations and seven seeds. On the prespecified harder combination, in-setting SB3 DQN had the primary return advantage despite the default-trained agent slicing more fruit and hitting more bombs. The [saved result](evidence/slicer_fresh_transfer_40960_results.json) records all four agent types and their limits.

`make gridquest-transfer` trains scratch PPO with egocentric or absolute views on 1, 10, 100 or 1,000 generated layouts and evaluates 200 disjoint test layouts, under a [protocol](evidence/gridquest_transfer_protocol.json) committed before training. The egocentric 1,000-layout model reached test SPL 0.9005 [0.8764, 0.9233] versus 0.0457 for random and 1.0 for map-knowing search. This is within one layout generator, and the remaining train-test gap is positive ([result](evidence/gridquest_transfer_results.json)). The full run takes about 29 minutes with eight workers; it is not part of `make all`.

`make unseen-protocol` freezes eight seeded held-out GridQuest maps (each with its own key, door, goal, walls and traps), plus the policies, seeds and metrics, in [a protocol file](evidence/unseen_levels_protocol.json), and refuses to overwrite it. `make unseen` refuses to run if the regenerated maps differ, then hunts every defect with every policy on every map and on clean copies. All four policies raised zero false alarms on the clean maps. Detection carried over for the random and coverage policies but fell for the count-bonus policy ([results](evidence/unseen_levels_results.json)).

The playtest comparison now stops when observed movement or score breaks the game's declared rules, without reading the injected defect label. A regression test also fixed the score-overflow scenario so it happens on a repeat coin visit, as intended. The [new 80-run evidence](evidence/playtest_contract_oracle.json) is separate from the earlier labeled-fixture experiment; [Results](RESULTS.md#contract-based-playtesting-18-october-2026) gives both tables and explains the change. The oracle still knows this small game's map and score bound, so unseen-game detection is open.

## Repository map

`envs/` defines the games; `agents/` implements the policies; `playtest/` injects defects; `evidence/` and `figures/` hold seeded outputs.

## Scope

The environments, layout generator and bug set are authored fixtures. Detection rates and transfer scores here do not estimate performance on commercial games. The comparison figure shows evaluation endpoints, and the 10,240-step run started fresh agents; the separate checkpoint-curve figure is the learning curve. The held-out maps are generated variations of one small game.
