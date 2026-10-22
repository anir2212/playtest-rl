"""Checkpoint learning curves: one evolving model per agent/seed, evaluated in place.

The protocol below was fixed on 20 October 2026 before the first curve run. Every
checkpoint evaluates the *same* network that continues training afterward; nothing
is stitched together from independently trained endpoints.
"""
from __future__ import annotations

import hashlib

import numpy as np
import torch

from agents import dqn, ppo
from analysis.aggregate import mean_ci

PROTOCOL = {
    "declared": "2026-10-20, before the first curve run",
    "agents": ["double_dqn", "scratch_ppo"],
    "training_seeds": [0, 1, 2, 3, 4],
    "total_steps": 20480,
    "checkpoint_every": 1024,
    "revision": "2026-10-20: spacing changed from 2,048 to 1,024 after the first run, because "
                "2,048 never lands on the historical 5,120-step endpoint used as a consistency "
                "check. Threshold, budget, seeds and evaluation are unchanged; spacing does not "
                "alter training (tests/test_curves.py).",
    # 3,584 = 0.7 x 5,120: the first 5,120 steps follow the historical 5,120-step
    # schedule, and a longer budget no longer stretches early exploration.
    "dqn_epsilon_decay_steps": 3584,
    "dqn_epsilon_start_final": [1.0, 0.05],
    "ppo_rollout_steps": 512,
    "eval_episodes": 30,
    "eval_seed_rule": "1000 + training seed; the same 30 Slicer episodes at every checkpoint",
    "eval_policy": "greedy (argmax), no sampling, so evaluation leaves training RNG untouched",
    "threshold_metric": "per-seed mean fruit score over the 30 evaluation episodes",
    "threshold": 5.0,
    "threshold_rule": "first checkpoint at or above the threshold; seeds that never reach it "
                      "are right-censored at total_steps",
}

AGENTS = {"double_dqn": (dqn.train, dqn.evaluate), "scratch_ppo": (ppo.train, ppo.evaluate)}


def fingerprint(model: torch.nn.Module) -> str:
    """Short SHA-256 of the parameters, to show checkpoints are one changing network."""
    digest = hashlib.sha256()
    for name, tensor in sorted(model.state_dict().items()):
        digest.update(name.encode())
        digest.update(tensor.detach().cpu().numpy().tobytes())
    return digest.hexdigest()[:16]


def train_with_checkpoints(agent: str, seed: int, steps: int, every: int,
                           eval_episodes: int, epsilon_decay_steps: int):
    """Train one agent from scratch and evaluate it at each checkpoint."""
    train, evaluate = AGENTS[agent]
    eval_seed = 1000 + seed
    checkpoints, model_ids = [], set()

    def record(step, model):
        model_ids.add(id(model))
        checkpoints.append({"step": int(step), "param_sha256_16": fingerprint(model),
                            **evaluate(model, seed=eval_seed, episodes=eval_episodes)})

    extra = {"epsilon_decay_steps": epsilon_decay_steps} if agent == "double_dqn" else {}
    model, log = train(seed, steps=steps, checkpoint_every=every, on_checkpoint=record, **extra)
    if model_ids != {id(model)}:
        raise RuntimeError("checkpoints did not come from the returned training run")
    return {"agent": agent, "seed": seed, "run_id": f"{agent}-seed{seed}-{steps}",
            "eval_seed": eval_seed, "eval_episodes": eval_episodes,
            "trained_steps": int(log["steps"]), "checkpoints": checkpoints}


def steps_to_threshold(checkpoints, threshold: float):
    """First checkpoint step reaching the threshold, or None when censored."""
    for row in checkpoints:
        if row["mean_score"] >= threshold:
            return row["step"]
    return None


def summarize(rows, threshold: float):
    summary = {}
    for agent in sorted({r["agent"] for r in rows}):
        runs = [r for r in rows if r["agent"] == agent]
        steps = [c["step"] for c in runs[0]["checkpoints"]]
        if any([c["step"] for c in r["checkpoints"]] != steps for r in runs):
            raise ValueError(f"{agent}: seeds have different checkpoint steps")
        if len({r["seed"] for r in runs}) != len(runs):
            raise ValueError(f"{agent}: duplicate training seeds")
        curve = []
        for index, step in enumerate(steps):
            values = [r["checkpoints"][index]["mean_score"] for r in runs]
            bombs = [r["checkpoints"][index]["mean_bomb_hits"] for r in runs]
            curve.append({"step": step, "mean_score": mean_ci(values),
                          "mean_bomb_hits": float(np.mean(bombs))})
        reached = {r["seed"]: steps_to_threshold(r["checkpoints"], threshold) for r in runs}
        summary[agent] = {"seeds": len(runs), "curve": curve,
                          "steps_to_threshold_by_seed": reached,
                          "reached_threshold": sum(v is not None for v in reached.values()),
                          "censored_at_steps": steps[-1]}
    return summary


def run_curve(seeds: int, steps: int, every: int, eval_episodes: int = 30,
              epsilon_decay_steps: int = PROTOCOL["dqn_epsilon_decay_steps"]):
    rows = [train_with_checkpoints(agent, seed, steps, every, eval_episodes, epsilon_decay_steps)
            for agent in PROTOCOL["agents"] for seed in range(seeds)]
    settings = {"training_seeds": list(range(seeds)), "total_steps": steps,
                "checkpoint_every": every, "eval_episodes": eval_episodes,
                "dqn_epsilon_decay_steps": epsilon_decay_steps}
    matches = all(settings[k] == PROTOCOL[k] for k in settings)
    return {"protocol": PROTOCOL, "settings": settings, "matches_protocol": matches,
            "rows": rows, "summary": summarize(rows, PROTOCOL["threshold"])}


def endpoint_consistency(curve: dict, endpoint: dict, agent: str, step: int):
    """Compare curve checkpoints with an independently saved endpoint run, seed by seed."""
    endpoint_rows = {r["seed"]: r["eval"] for r in endpoint["rows"] if r["agent"] == agent}
    pairs = []
    for row in curve["rows"]:
        if row["agent"] != agent or row["seed"] not in endpoint_rows:
            continue
        checkpoint = next((c for c in row["checkpoints"] if c["step"] == step), None)
        if checkpoint is None:
            raise ValueError(f"no checkpoint at step {step}")
        pairs.append({"seed": row["seed"], "curve": checkpoint["mean_score"],
                      "endpoint": endpoint_rows[row["seed"]]["mean_score"]})
    return {"agent": agent, "step": step, "pairs": pairs,
            "identical": bool(pairs) and all(p["curve"] == p["endpoint"] for p in pairs)}


def consistency_report(curve_path, endpoints):
    """Seed-level comparison of curve checkpoints with saved independent endpoint runs."""
    import json
    from pathlib import Path
    curve = json.loads(Path(curve_path).read_text())
    checks = []
    for endpoint_path, step in endpoints:
        endpoint = json.loads(Path(endpoint_path).read_text())
        for agent in curve["protocol"]["agents"]:
            checks.append({"endpoint_file": str(endpoint_path),
                           **endpoint_consistency(curve, endpoint, agent, step)})
    return {"curve_file": str(curve_path), "checks": checks}



if __name__ == "__main__":
    import json
    from pathlib import Path
    # Relative paths keep the saved report portable; run from the project root.
    curve_file = Path("evidence/learning_curve_20480.json")
    if not curve_file.exists():
        raise SystemExit(f"{curve_file} not found; run `make curve` and copy it, from the project root")
    report = consistency_report(curve_file, [(Path("evidence/neural.json"), 5120),
                                             (Path("evidence/neural_10240.json"), 10240)])
    Path("evidence/learning_curve_consistency.json").write_text(json.dumps(report, indent=2) + "\n")
    print({(c["agent"], c["step"]): c["identical"] for c in report["checks"]})
