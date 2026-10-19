# PlaytestRL

PlaytestRL uses two small Gymnasium games to ask how an automated player learns and whether it can expose a broken game rule. GridQuest makes reward-design mistakes visible; Slicer supports a bounded comparison of hand-written and library agents. A separate playtest harness measures how four search policies find planted defects.

![Seeded agent endpoint comparison](figures/learning_comparison.png)

## Why this project

Manual game testing can miss paths that nobody thought to script. An agent may reach those paths, but it may also learn to farm a badly designed reward. The games and injected defects here are small enough to inspect, so both successes and failures have an explanation.

## Current milestones

| Area | What works today | Next question |
|---|---|---|
| Games and agents | Checked GridQuest/Slicer environments, tabular Q-learning, scratch Double DQN/PPO and library references | What changes on unseen levels? |
| Reward and bug studies | A reward-farming counterexample and a contract oracle for four planted defects | Can the oracle transfer to another game contract? |
| Training evidence | Separate five-seed 5,120- and 10,240-step endpoints | What does a true checkpoint curve show under a fixed exploration schedule? |

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

The playtest comparison now stops when observed movement or score breaks the game's declared rules, without reading the injected defect label. A regression test also fixed the score-overflow scenario so it happens on a repeat coin visit, as intended. The [new 80-run evidence](evidence/playtest_contract_oracle.json) is separate from the earlier labeled-fixture experiment; [Results](RESULTS.md#contract-based-playtesting-18-october-2026) gives both tables and explains the change. The oracle still knows this small game's map and score bound, so unseen-game detection is open.

## Repository map

`envs/` defines the games; `agents/` implements the policies; `playtest/` injects defects; `evidence/` and `figures/` hold seeded outputs.

## Scope

The environments and bug set are authored fixtures. Detection rates here do not estimate performance on commercial games. The comparison figure shows evaluation endpoints rather than a learning curve, and the longer run started fresh agents.
