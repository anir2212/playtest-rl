"""Predeclared Slicer difficulty sweep and held-out-setting evaluation for the learning agents.

Two steps, run in order and never merged:

1. ``--write-protocol`` copies ``PROTOCOL`` (difficulty settings, which ones are held
   out, seeds, budget, anchors and metrics) to ``evidence/slicer_difficulty_protocol.json``
   with the SHA-256 of the curve evidence it must reproduce. It refuses to overwrite.
2. ``--run`` refuses to start if that file no longer matches ``PROTOCOL`` or the curve
   evidence changed. It then trains

   * the **training-setting models**: scratch Double DQN and PPO, seeds 0-4, on the
     default Slicer under the frozen 20 October curve protocol, evaluated at every
     checkpoint on the training setting and on every held-out setting; and
   * the **in-setting references** (the difficulty sweep): the same agents, seeds and
     budget trained directly on each held-out setting and evaluated on it,

   and evaluates a random and a scripted anchor policy on every setting. The result is
   written to ``runs/difficulty.json``; the committed copy of the one predeclared run is
   ``evidence/slicer_difficulty_20480.json``.

Held-out settings never choose a hyperparameter, policy, anchor or checkpoint. Every
setting is fully observable through the existing six-value observation; only the
object speed and the bomb share change.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import multiprocessing
from concurrent.futures import ProcessPoolExecutor
from datetime import UTC, datetime
from pathlib import Path

import numpy as np

from analysis import curves
from analysis.aggregate import mean_ci
from envs.slicer import Slicer

ROOT = Path(__file__).resolve().parents[1]
PROTOCOL_PATH = ROOT / "evidence" / "slicer_difficulty_protocol.json"
EVIDENCE_PATH = ROOT / "evidence" / "slicer_difficulty_20480.json"
CURVE_EVIDENCE = ROOT / "evidence" / "learning_curve_20480.json"
RUN_PATH = ROOT / "runs" / "difficulty.json"
REQUIRED_SEEDS = 5

PROTOCOL = {
    "name": "slicer-difficulty-v1",
    "declared": "2026-10-22, before any agent was trained or evaluated on a held-out setting",
    "agents": ["double_dqn", "scratch_ppo"],
    "excluded_agents": "Tabular Q (its GridQuest state is a map-specific position index and "
                       "cannot transfer by construction) and the SB3 references (no curve "
                       "protocol exists for them yet).",
    "training_seeds": [0, 1, 2, 3, 4],
    "training_setting": "train",
    "settings": {
        "train": {"bomb_ratio": 0.2, "speed": 1.0},
        "no_bombs": {"bomb_ratio": 0.0, "speed": 1.0},
        "slow_0.7": {"bomb_ratio": 0.2, "speed": 0.7},
        "bombs_0.4": {"bomb_ratio": 0.4, "speed": 1.0},
        "bombs_0.6": {"bomb_ratio": 0.6, "speed": 1.0},
        "fast_1.3": {"bomb_ratio": 0.2, "speed": 1.3},
        "fast_1.6": {"bomb_ratio": 0.2, "speed": 1.6},
        "bombs_0.4_fast_1.3": {"bomb_ratio": 0.4, "speed": 1.3},
    },
    "held_out": ["no_bombs", "slow_0.7", "bombs_0.4", "bombs_0.6", "fast_1.3", "fast_1.6",
                 "bombs_0.4_fast_1.3"],
    "setting_notes": "speed scales both sampled object velocities (default range: vertical "
                     "0.03-0.055, horizontal +/-0.015 per tick); bomb_ratio is the chance a "
                     "spawned object is a bomb. Horizon (80), cursor step (0.12), slice "
                     "window (0.14) and rewards are unchanged.",
    "training": "Frozen 20 October curve protocol (analysis/curves.py PROTOCOL): 20,480 "
                "environment steps, checkpoints every 1,024 steps, DQN epsilon 1.0 to 0.05 "
                "over 3,584 absolute steps, PPO 512-step rollouts, every other "
                "hyperparameter at its code default. Nothing is tuned for this protocol.",
    "total_steps": 20480,
    "checkpoint_every": 1024,
    "dqn_epsilon_decay_steps": 3584,
    "eval_episodes": 30,
    "eval_seed_base": 7000,
    "eval_seed_rule": "7000 + training seed on every setting, a seed block no earlier run "
                      "used; episode e resets with seed (7000 + training seed) * 100000 + e",
    "consistency_eval_seed_base": 1000,
    "consistency_rule": "Training-setting models are also evaluated on the default Slicer "
                        "with evaluation seed 1000 + training seed at every checkpoint. "
                        "Those 210 rows must equal evidence/learning_curve_20480.json "
                        "(score, bomb hits, return and parameter fingerprint).",
    "eval_policy": "greedy (argmax), as in the curve protocol",
    "anchors": {
        "random": "uniform random action, numpy default_rng(evaluation seed)",
        "scripted": "fixed rule written before any held-out evaluation: stay on a bomb; on a "
                    "fruit, slice if next-tick y > 0.2 and |cursor - next-tick x| < 0.14, "
                    "otherwise step toward next-tick x when more than 0.06 away",
    },
    "primary_metric": "Per-seed mean fruit score over the 30 evaluation episodes at 20,480 "
                      "steps, for training-setting models on each setting; five-seed mean "
                      "with the seed-level bootstrap 95% interval (analysis.aggregate). "
                      "Mean bomb hits reported alongside.",
    "secondary_metrics": [
        "anchor-normalized score per seed, (agent - random) / (scripted - random), using the "
        "anchors on the same evaluation episodes; defined for a setting only if scripted "
        "beats random by more than 0.5 fruit on every evaluation seed",
        "transfer gap = in-setting reference mean minus training-setting model mean on that "
        "held-out setting at 20,480 steps, with a 2,000-resample bootstrap over the two "
        "independent five-seed groups",
        "held-out score at every 1,024-step checkpoint, descriptive only; no checkpoint is "
        "selected as best",
    ],
    "decision_rule": "For each agent and held-out setting: 'transfer loss' if the gap's 95% "
                     "interval lies above 0, 'transfer gain' if below 0, otherwise 'no "
                     "measurable difference' (with five seeds this is not evidence of "
                     "equivalence). Every result is reported, whatever its direction.",
    "tuning": "None. Settings, seeds, budgets, anchors and metrics are frozen by this "
              "protocol; misses and failures are reported, not re-run.",
}


# ---------------------------------------------------------------- protocol guards

def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def check_protocol(protocol: dict) -> None:
    """Structural rules every protocol (including test protocols) must satisfy."""
    seeds = protocol["training_seeds"]
    if len(seeds) != REQUIRED_SEEDS or len(set(seeds)) != REQUIRED_SEEDS:
        raise ValueError(f"need {REQUIRED_SEEDS} distinct training seeds, got {seeds}")
    settings, train = protocol["settings"], protocol["training_setting"]
    if train not in settings or train in protocol["held_out"]:
        raise ValueError("the training setting must be declared and not held out")
    if set(protocol["held_out"]) != set(settings) - {train}:
        raise ValueError("every non-training setting must be listed as held out")
    if len({tuple(sorted(v.items())) for v in settings.values()}) != len(settings):
        raise ValueError("two settings are identical")
    if protocol["eval_seed_base"] == protocol["consistency_eval_seed_base"]:
        raise ValueError("held-out evaluation must use a fresh seed block")


def write_protocol(path: Path = PROTOCOL_PATH, protocol: dict = PROTOCOL,
                   curve_evidence: Path = CURVE_EVIDENCE) -> dict:
    if path.exists():
        raise FileExistsError(f"{path} exists; a predeclared protocol is never overwritten")
    check_protocol(protocol)
    body = {**protocol, "written_at_utc": datetime.now(UTC).isoformat(timespec="seconds"),
            "curve_evidence_sha256": sha256(curve_evidence)}
    path.write_text(json.dumps(body, indent=2) + "\n", encoding="utf-8")
    return body


def load_protocol(path: Path = PROTOCOL_PATH, protocol: dict = PROTOCOL,
                  curve_evidence: Path = CURVE_EVIDENCE) -> dict:
    """Read the frozen file and refuse if it drifted from the constant or its inputs."""
    saved = json.loads(path.read_text(encoding="utf-8"))
    declared = {k: v for k, v in saved.items()
                if k not in ("written_at_utc", "curve_evidence_sha256")}
    if declared != json.loads(json.dumps(protocol)):
        raise RuntimeError(f"{path} differs from the PROTOCOL constant; refusing to run")
    if curve_evidence.exists() and sha256(curve_evidence) != saved["curve_evidence_sha256"]:
        raise RuntimeError("curve evidence changed since the protocol was written")
    check_protocol(saved)
    return saved


# ---------------------------------------------------------------- anchors

def scripted_action(obs) -> int:
    cursor, x, y, vx, vy, bomb = (float(v) for v in obs)
    if bomb >= 0.5:
        return 0
    next_x, next_y = x + vx, y + vy
    if next_y > 0.2 and abs(cursor - next_x) < 0.14:
        return 3
    if next_x < cursor - 0.06:
        return 1
    if next_x > cursor + 0.06:
        return 2
    return 0


def evaluate_anchor(policy: str, seed: int, episodes: int, bomb_ratio: float, speed: float):
    """Same episode seeds and metrics as the agents' ``evaluate`` functions."""
    env = Slicer(bomb_ratio=bomb_ratio, speed=speed)
    rng = np.random.default_rng(seed)
    scores, bombs, returns = [], [], []
    for episode in range(episodes):
        obs, _ = env.reset(seed=seed * 100000 + episode)
        total = 0.0
        for _ in range(env.horizon):
            action = int(rng.integers(4)) if policy == "random" else scripted_action(obs)
            obs, reward, terminated, truncated, info = env.step(action)
            total += reward
            if terminated or truncated:
                break
        scores.append(info["score"])
        bombs.append(info["bomb_hits"])
        returns.append(total)
    env.close()
    return {"mean_score": float(np.mean(scores)), "mean_bomb_hits": float(np.mean(bombs)),
            "mean_return": float(np.mean(returns))}


# ---------------------------------------------------------------- training jobs

def train_job(job: dict) -> dict:
    """Train one agent on one setting; evaluate the same evolving model at every checkpoint."""
    protocol, agent, seed = job["protocol"], job["agent"], job["seed"]
    settings = protocol["settings"]
    train, evaluate = curves.AGENTS[agent]
    eval_seed = protocol["eval_seed_base"] + seed
    consistency_seed = protocol["consistency_eval_seed_base"] + seed
    episodes = protocol["eval_episodes"]
    checkpoints, model_ids = [], set()

    def record(step, model):
        model_ids.add(id(model))
        row = {"step": int(step), "param_sha256_16": curves.fingerprint(model),
               "eval": {name: evaluate(model, seed=eval_seed, episodes=episodes, **settings[name])
                        for name in job["eval_settings"]}}
        if job["consistency"]:
            row["consistency_eval"] = evaluate(model, seed=consistency_seed, episodes=episodes)
        checkpoints.append(row)

    extra = ({"epsilon_decay_steps": protocol["dqn_epsilon_decay_steps"]}
             if agent == "double_dqn" else {})
    model, log = train(seed, steps=protocol["total_steps"],
                       checkpoint_every=protocol["checkpoint_every"], on_checkpoint=record,
                       **settings[job["train_setting"]], **extra)
    if model_ids != {id(model)}:
        raise RuntimeError("checkpoints did not come from the returned training run")
    return {"agent": agent, "train_setting": job["train_setting"], "seed": seed,
            "eval_seed": eval_seed,
            "consistency_eval_seed": consistency_seed if job["consistency"] else None,
            "trained_steps": int(log["steps"]), "checkpoints": checkpoints}


def build_jobs(protocol: dict) -> list[dict]:
    train, everything = protocol["training_setting"], list(protocol["settings"])
    jobs = [{"protocol": protocol, "agent": agent, "seed": seed, "train_setting": train,
             "eval_settings": everything, "consistency": True}
            for agent in protocol["agents"] for seed in protocol["training_seeds"]]
    jobs += [{"protocol": protocol, "agent": agent, "seed": seed, "train_setting": held,
              "eval_settings": [held], "consistency": False}
             for held in protocol["held_out"] for agent in protocol["agents"]
             for seed in protocol["training_seeds"]]
    return jobs


# ---------------------------------------------------------------- summaries

def gap_ci(in_setting, transferred, seed: int = 91, resamples: int = 2000) -> dict:
    """Bootstrap the difference of two independent seed-mean groups."""
    a, b = np.asarray(in_setting, dtype=float), np.asarray(transferred, dtype=float)
    rng = np.random.default_rng(seed)
    draws = (a[rng.integers(0, len(a), size=(resamples, len(a)))].mean(axis=1)
             - b[rng.integers(0, len(b), size=(resamples, len(b)))].mean(axis=1))
    low, high = float(np.quantile(draws, 0.025)), float(np.quantile(draws, 0.975))
    reading = ("transfer loss" if low > 0 else "transfer gain" if high < 0
               else "no measurable difference")
    return {"mean": float(a.mean() - b.mean()), "ci95": [low, high], "reading": reading}


def final_values(rows, agent, train_setting, eval_setting, metric="mean_score"):
    """Per-seed values at the last checkpoint, ordered by seed; exactly five distinct seeds."""
    group = sorted((r for r in rows if r["agent"] == agent and r["train_setting"] == train_setting),
                   key=lambda r: r["seed"])
    seeds = [r["seed"] for r in group]
    if len(seeds) != REQUIRED_SEEDS or len(set(seeds)) != REQUIRED_SEEDS:
        raise ValueError(f"{agent}/{train_setting}: expected {REQUIRED_SEEDS} distinct seeds, got {seeds}")
    if any(r["checkpoints"][-1]["step"] != r["trained_steps"] for r in group):
        raise ValueError("last checkpoint is not the end of training")
    return [r["checkpoints"][-1]["eval"][eval_setting][metric] for r in group]


def curve_points(group: list[dict], setting: str) -> list[dict]:
    """Five-seed mean and bootstrap interval at every checkpoint, for one evaluation setting."""
    steps = [c["step"] for c in group[0]["checkpoints"]]
    if any([c["step"] for c in r["checkpoints"]] != steps for r in group):
        raise ValueError("seeds have different checkpoint steps")
    points = []
    for index, step in enumerate(steps):
        ci = mean_ci([r["checkpoints"][index]["eval"][setting]["mean_score"] for r in group])
        points.append({"step": step, "mean": ci["mean"], "low": ci["ci95"][0], "high": ci["ci95"][1]})
    return points


def summarize(rows: list[dict], anchors: list[dict], protocol: dict) -> dict:
    train = protocol["training_setting"]
    anchor = {}
    for setting in protocol["settings"]:
        anchor[setting] = {}
        for policy in ("random", "scripted"):
            group = sorted((a for a in anchors if a["setting"] == setting and a["policy"] == policy),
                           key=lambda a: a["seed"])
            anchor[setting][policy] = {m: mean_ci([a[m] for a in group])
                                       for m in ("mean_score", "mean_bomb_hits")}
    summary = {"anchors": anchor, "agents": {}}
    for agent in protocol["agents"]:
        per_setting = {}
        for setting in protocol["settings"]:
            scores = final_values(rows, agent, train, setting)
            bombs = final_values(rows, agent, train, setting, "mean_bomb_hits")
            rand = anchor[setting]["random"]["mean_score"]["seed_values"]
            script = anchor[setting]["scripted"]["mean_score"]["seed_values"]
            spans = [s - r for s, r in zip(script, rand)]
            normalized = (mean_ci([(x - r) / span for x, r, span in zip(scores, rand, spans)])
                          if min(spans) > 0.5 else None)
            entry = {"mean_score": mean_ci(scores), "mean_bomb_hits": mean_ci(bombs),
                     "normalized_score": normalized}
            if setting != train:
                own = final_values(rows, agent, setting, setting)
                own_bombs = final_values(rows, agent, setting, setting, "mean_bomb_hits")
                entry["in_setting_reference"] = {"mean_score": mean_ci(own),
                                                 "mean_bomb_hits": mean_ci(own_bombs)}
                entry["transfer_gap"] = gap_ci(own, scores)
            per_setting[setting] = entry
        curve = {}
        for setting in protocol["settings"]:
            for train_setting in ([train] if setting == train else [train, setting]):
                group = [r for r in rows if r["agent"] == agent and r["train_setting"] == train_setting]
                curve[f"{train_setting}->{setting}"] = curve_points(group, setting)
        summary["agents"][agent] = {"final_step": protocol["total_steps"], "settings": per_setting,
                                    "checkpoint_curves": curve}
    return summary


def curve_consistency(rows: list[dict], curve_evidence: Path = CURVE_EVIDENCE) -> dict:
    """Compare the training-setting checkpoints (seed 1000 + s) with the committed curve run."""
    saved = json.loads(curve_evidence.read_text(encoding="utf-8"))
    reference = {(r["agent"], r["seed"]): r["checkpoints"] for r in saved["rows"]}
    compared = matched = 0
    mismatches = []
    for row in rows:
        if row["consistency_eval_seed"] is None:
            continue
        for new, old in zip(row["checkpoints"], reference[(row["agent"], row["seed"])], strict=True):
            compared += 1
            same = (new["step"] == old["step"] and new["param_sha256_16"] == old["param_sha256_16"]
                    and all(new["consistency_eval"][k] == old[k]
                            for k in ("mean_score", "mean_bomb_hits", "mean_return")))
            matched += same
            if not same:
                mismatches.append({"agent": row["agent"], "seed": row["seed"], "step": new["step"]})
    return {"curve_file": str(curve_evidence.relative_to(ROOT)) if curve_evidence.is_relative_to(ROOT)
            else str(curve_evidence), "checkpoints_compared": compared,
            "checkpoints_identical": matched, "identical": compared > 0 and matched == compared,
            "mismatches": mismatches[:20]}


# ---------------------------------------------------------------- run

def run(protocol_path: Path = PROTOCOL_PATH, out: Path = RUN_PATH, workers: int = 1,
        protocol_constant: dict = PROTOCOL, curve_evidence: Path = CURVE_EVIDENCE,
        compare_with: Path | None = EVIDENCE_PATH) -> dict:
    if not 1 <= workers <= 4:
        raise ValueError("use between 1 and 4 worker processes")
    protocol = load_protocol(protocol_path, protocol_constant, curve_evidence)
    started = datetime.now(UTC)
    jobs = build_jobs(protocol)
    if workers == 1:
        rows = [train_job(job) for job in jobs]
    else:
        context = multiprocessing.get_context("spawn")
        with ProcessPoolExecutor(max_workers=workers, mp_context=context) as pool:
            rows = list(pool.map(train_job, jobs))
    anchors = [{"setting": name, "policy": policy, "seed": seed,
                "eval_seed": protocol["eval_seed_base"] + seed,
                **evaluate_anchor(policy, protocol["eval_seed_base"] + seed,
                                  protocol["eval_episodes"], **kwargs)}
               for name, kwargs in protocol["settings"].items()
               for policy in ("random", "scripted") for seed in protocol["training_seeds"]]
    comparable = all(protocol[k] == curves.PROTOCOL[k] for k in
                     ("training_seeds", "total_steps", "checkpoint_every",
                      "dqn_epsilon_decay_steps", "eval_episodes"))
    result = {
        "protocol": protocol["name"], "protocol_file": protocol_path.name,
        "protocol_sha256": sha256(protocol_path),
        "protocol_written_at_utc": protocol["written_at_utc"],
        "started_at_utc": started.isoformat(timespec="seconds"),
        "finished_at_utc": datetime.now(UTC).isoformat(timespec="seconds"),
        "workers": workers,
        "curve_consistency": (curve_consistency(rows, curve_evidence) if comparable
                              else {"identical": None, "reason": "settings differ from the curve protocol"}),
        "summary": summarize(rows, anchors, protocol),
        "anchors": anchors, "rows": rows,
    }
    if compare_with is not None and compare_with.exists():
        saved = json.loads(compare_with.read_text(encoding="utf-8"))
        result["matches_committed_evidence"] = (saved["rows"] == json.loads(json.dumps(rows))
                                                and saved["anchors"] == json.loads(json.dumps(anchors)))
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    return result


def headline(result: dict) -> dict:
    table = {}
    for agent, data in result["summary"]["agents"].items():
        for setting, entry in data["settings"].items():
            cell = {"score": round(entry["mean_score"]["mean"], 3)}
            if "transfer_gap" in entry:
                cell["in_setting"] = round(entry["in_setting_reference"]["mean_score"]["mean"], 3)
                cell["gap"] = entry["transfer_gap"]["reading"]
            table[f"{agent}|{setting}"] = cell
    return {"curve_consistency_identical": result["curve_consistency"]["identical"],
            "matches_committed_evidence": result.get("matches_committed_evidence"),
            "final_step": table}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    action = parser.add_mutually_exclusive_group(required=True)
    action.add_argument("--write-protocol", action="store_true")
    action.add_argument("--run", action="store_true")
    parser.add_argument("--workers", type=int, default=1, help="1-4 worker processes")
    args = parser.parse_args()
    if args.write_protocol:
        body = write_protocol()
        print(json.dumps({"written": str(PROTOCOL_PATH.relative_to(ROOT)),
                          "held_out": body["held_out"]}, indent=2))
        return
    result = run(workers=args.workers)
    print(f"{RUN_PATH.relative_to(ROOT)} written")
    print(json.dumps(headline(result), indent=2))


if __name__ == "__main__":
    main()
