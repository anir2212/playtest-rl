# PlaytestRL

Reinforcement-learning agents for repeatable game testing.

Custom Gymnasium game environments and agents for repeatable playtesting. The project compares tabular and neural agents, then uses seeded bug injections to measure what the agents catch.

![Seeded agent learning comparison](figures/learning_comparison.png)

## Why this project

Manual game testing misses rare paths, while scripted bots repeat only paths someone anticipated. This project tests whether learning agents can reach more states and expose seeded defects in small, repeatable environments.

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

The environments and bug set are authored fixtures. Detection rates here do not estimate performance on commercial games.
