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

## Repository map

`envs/` defines the games; `agents/` implements the policies; `playtest/` injects defects; `evidence/` and `figures/` hold seeded outputs.

## Scope

The environments and bug set are authored fixtures. Detection rates here do not estimate performance on commercial games.
