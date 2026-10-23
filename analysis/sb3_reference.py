"""Predeclared SB3 reference curves and held-out Slicer evaluation (``slicer-sb3-reference-v1``).

This puts the Stable-Baselines3 DQN and PPO references on the same footing as the
scratch agents, without touching either earlier protocol:

* the **checkpoint curve** of ``analysis/curves.py`` (20,480 steps, a checkpoint every
  1,024 steps, evaluation seed 1000 + training seed, threshold 5.0), taken through an
  SB3 callback on one evolving model; and
* the **held-out difficulty evaluation** of ``analysis/difficulty.py`` (the same eight
  Slicer settings, evaluation seed 7000 + training seed, in-setting references and the
  same random and scripted anchors).

Two steps, run in order and never merged:

1. ``--write-protocol`` copies ``PROTOCOL`` to ``evidence/sb3_reference_protocol.json``
   together with the SHA-256 of every frozen input it compares against. It refuses to
   overwrite.
2. ``--run`` refuses to start if that file differs from ``PROTOCOL`` or a frozen input
   changed, trains 80 SB3 models (2 agents x 5 seeds on the default setting plus on each
   of the 7 held-out settings) and writes ``runs/sb3_reference.json``. The committed
   copy of the one predeclared run is ``evidence/sb3_reference_20480.json``.

The frozen ``slicer-difficulty-v1`` protocol, the 20,480-step curve protocol and their
evidence are only read.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import multiprocessing
from concurrent.futures import ProcessPoolExecutor
from datetime import UTC, datetime
from functools import partial
from pathlib import Path

import torch
from stable_baselines3 import DQN as SB3DQN, PPO as SB3PPO
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.vec_env import DummyVecEnv

from analysis import curves, difficulty
from envs.slicer import Slicer
from experiments.run import eval_sb3

ROOT = Path(__file__).resolve().parents[1]
PROTOCOL_PATH = ROOT / "evidence" / "sb3_reference_protocol.json"
EVIDENCE_PATH = ROOT / "evidence" / "sb3_reference_20480.json"
RUN_PATH = ROOT / "runs" / "sb3_reference.json"
REQUIRED_SEEDS = 5
FROZEN_INPUTS = {
    "curve_evidence": "evidence/learning_curve_20480.json",
    "difficulty_protocol": "evidence/slicer_difficulty_protocol.json",
    "difficulty_evidence": "evidence/slicer_difficulty_20480.json",
    "endpoint_5120": "evidence/neural.json",
    "endpoint_10240": "evidence/neural_10240.json",
}
SCRATCH_TWIN = {"sb3_dqn": "double_dqn", "sb3_ppo": "scratch_ppo"}

PROTOCOL = {
    "name": "slicer-sb3-reference-v1",
    "declared": "2026-10-23, before any SB3 model was checkpointed, trained for 20,480 steps "
                "or evaluated on a held-out setting",
    "agents": ["sb3_dqn", "sb3_ppo"],
    "hyperparameters": {
        "sb3_dqn": {"learning_rate": 5e-4, "buffer_size": 10000, "learning_starts": 100,
                    "batch_size": 64, "train_freq": 4, "gradient_steps": 1,
                    "target_update_interval": 250, "exploration_initial_eps": 1.0,
                    "exploration_final_eps": 0.05, "gamma": 0.99},
        "sb3_ppo": {"learning_rate": 3e-4, "n_steps": 128, "batch_size": 128, "n_epochs": 4,
                    "gamma": 0.99, "gae_lambda": 0.95, "clip_range": 0.2, "ent_coef": 0.01},
    },
    "hyperparameter_source": "Identical to the SB3 references in experiments/run.py run_neural "
                             "(MlpPolicy, cpu, seed = training seed, PPO on 4 DummyVecEnv "
                             "copies), except that DQN's exploration fraction is set from "
                             "absolute steps as below. Nothing is tuned.",
    "training_seeds": [0, 1, 2, 3, 4],
    "total_steps": 20480,
    "checkpoint_every": 1024,
    "dqn_epsilon_decay_steps": 3584,
    "sb3_dqn_exploration_fraction": 0.175,
    "epsilon_rule": "SB3 defines exploration_fraction against total_timesteps, so it is set to "
                    "3,584 / 20,480 = 0.175: epsilon falls from 1.0 to 0.05 over the same 3,584 "
                    "absolute steps as the scratch curve and the 5,120-step endpoint run "
                    "(0.7 x 5,120). PPO has no budget-dependent schedule.",
    "ppo_rollout_steps": 512,
    "checkpoint_rule": "A stable_baselines3 callback evaluates the model at the start of every "
                       "rollout whose num_timesteps is a multiple of 1,024 (after all gradient "
                       "updates for earlier steps, as in the scratch trainers) and once more "
                       "after learn() returns: 21 checkpoints at 0, 1,024, ..., 20,480 on the "
                       "same evolving model. Evaluation is greedy (predict deterministic=True), "
                       "which draws no training random numbers.",
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
    "settings_source": "Copied unchanged from slicer-difficulty-v1; held-out settings choose "
                       "nothing.",
    "eval_episodes": 30,
    "curve_eval_seed_base": 1000,
    "eval_seed_base": 7000,
    "eval_seed_rule": "Curve: 1000 + training seed on the default setting at every checkpoint, "
                      "the scratch curve's 30 episodes. Held-out evaluation: 7000 + training "
                      "seed on every setting, the scratch difficulty run's episodes. Episode e "
                      "resets with seed (base + training seed) * 100000 + e.",
    "anchors": "Random and scripted anchors recomputed with analysis.difficulty.evaluate_anchor "
               "on the same episodes; they must equal the anchors in "
               "evidence/slicer_difficulty_20480.json.",
    "primary_metrics": [
        "Curve: per-seed mean fruit score at each of the 21 checkpoints (evaluation seed "
        "1000 + s), five-seed mean with the seed-level bootstrap 95% interval; mean bomb "
        "hits alongside.",
        "Curve: first checkpoint at or above mean fruit score 5.0 per seed; seeds that never "
        "reach it are right-censored at 20,480 and never assigned that step.",
        "Held-out: per-seed mean fruit score at 20,480 steps for the default-trained models on "
        "each setting (evaluation seed 7000 + s), seed bootstrap interval, bomb hits alongside.",
    ],
    "secondary_metrics": [
        "anchor-normalized score and transfer gap (in-setting reference minus default-trained "
        "model) exactly as in slicer-difficulty-v1",
        "implementation gap = SB3 mean minus scratch twin mean at 20,480 steps (sb3_dqn vs "
        "double_dqn, sb3_ppo vs scratch_ppo), on the curve episodes and on every setting for "
        "the default-trained models, with the same 2,000-resample bootstrap over the two "
        "five-seed groups; scratch values are read from the committed evidence",
        "mean episode return reported beside fruit score, descriptive only",
        "held-out score at every checkpoint, descriptive only; no checkpoint is selected",
    ],
    "decision_rules": {
        "transfer_gap": "'transfer loss' if the 95% interval lies above 0, 'transfer gain' if "
                        "below 0, otherwise 'no measurable difference' (not evidence of "
                        "equivalence with five seeds)",
        "implementation_gap": "'SB3 higher' if the 95% interval lies above 0, 'scratch higher' "
                              "if below 0, otherwise 'no measurable difference'. Descriptive: "
                              "network layout, time-limit bootstrapping and optimizer details "
                              "differ between the implementations.",
    },
    "consistency_predictions": [
        "The 5,120-step checkpoint equals the independently trained 5,120-step SB3 endpoint in "
        "evidence/neural.json seed for seed, for both agents (same seeds, hyperparameters and "
        "a 3,584-step epsilon decay).",
        "SB3 PPO's 10,240-step checkpoint equals evidence/neural_10240.json seed for seed.",
        "SB3 DQN's 10,240-step checkpoint differs from evidence/neural_10240.json, because "
        "that run decayed epsilon over 0.7 x 10,240 = 7,168 steps.",
        "A failed prediction is reported as a finding about the callback path; nothing is "
        "re-run or changed because of it.",
    ],
    "tuning": "None. Agents, hyperparameters, settings, seeds, budget, anchors, metrics and "
              "reading rules are frozen by this protocol; failures are reported, not re-run.",
}


# ---------------------------------------------------------------- protocol guards

def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def check_protocol(protocol: dict) -> None:
    """Structural rules every protocol (including test protocols) must satisfy."""
    seeds = protocol["training_seeds"]
    if len(seeds) != REQUIRED_SEEDS or len(set(seeds)) != REQUIRED_SEEDS:
        raise ValueError(f"need {REQUIRED_SEEDS} distinct training seeds, got {seeds}")
    if set(protocol["agents"]) != set(SCRATCH_TWIN):
        raise ValueError("the protocol covers exactly the two SB3 references")
    total, decay = protocol["total_steps"], protocol["dqn_epsilon_decay_steps"]
    if not 0 < decay <= total or protocol["sb3_dqn_exploration_fraction"] != decay / total:
        raise ValueError("exploration fraction must equal epsilon decay steps / total steps")
    every = protocol["checkpoint_every"]
    if every <= 0 or total % every or every % protocol["ppo_rollout_steps"]:
        raise ValueError("checkpoints must divide the budget and be whole PPO rollouts")
    settings, train = protocol["settings"], protocol["training_setting"]
    if train not in settings or train in protocol["held_out"]:
        raise ValueError("the training setting must be declared and not held out")
    if set(protocol["held_out"]) != set(settings) - {train}:
        raise ValueError("every non-training setting must be listed as held out")
    if protocol["eval_seed_base"] == protocol["curve_eval_seed_base"]:
        raise ValueError("held-out evaluation must use its own seed block")


def write_protocol(path: Path = PROTOCOL_PATH, protocol: dict = PROTOCOL,
                   root: Path = ROOT) -> dict:
    if path.exists():
        raise FileExistsError(f"{path} exists; a predeclared protocol is never overwritten")
    check_protocol(protocol)
    body = {**protocol, "written_at_utc": datetime.now(UTC).isoformat(timespec="seconds"),
            "frozen_inputs_sha256": {name: sha256(root / rel) for name, rel in FROZEN_INPUTS.items()}}
    path.write_text(json.dumps(body, indent=2) + "\n", encoding="utf-8")
    return body


def load_protocol(path: Path = PROTOCOL_PATH, protocol: dict = PROTOCOL,
                  root: Path = ROOT) -> dict:
    """Read the frozen file and refuse if it drifted from the constant or its inputs."""
    saved = json.loads(path.read_text(encoding="utf-8"))
    declared = {k: v for k, v in saved.items()
                if k not in ("written_at_utc", "frozen_inputs_sha256")}
    if declared != json.loads(json.dumps(protocol)):
        raise RuntimeError(f"{path} differs from the PROTOCOL constant; refusing to run")
    for name, rel in FROZEN_INPUTS.items():
        if sha256(root / rel) != saved["frozen_inputs_sha256"][name]:
            raise RuntimeError(f"{rel} changed since the protocol was written")
    check_protocol(saved)
    return saved


# ---------------------------------------------------------------- training with checkpoints

class Checkpoints(BaseCallback):
    """Call ``record(step, model)`` at rollout starts on multiples of ``every``.

    A rollout starts only after the previous rollout's gradient updates, so the model
    seen at step k has learned from every transition before k, as in the scratch agents.
    """

    def __init__(self, every: int, record):
        super().__init__()
        self.every, self.record, self.steps = every, record, []

    def _on_rollout_start(self) -> None:
        step = int(self.model.num_timesteps)
        if step % self.every == 0 and step not in self.steps:
            self.steps.append(step)
            self.record(step, self.model)

    def _on_step(self) -> bool:
        return True


def build_model(agent: str, seed: int, protocol: dict, bomb_ratio: float, speed: float):
    torch.set_num_threads(2)
    torch.manual_seed(seed)
    hyper = protocol["hyperparameters"][agent]
    if agent == "sb3_dqn":
        env = Slicer(bomb_ratio=bomb_ratio, speed=speed)
        model = SB3DQN("MlpPolicy", env, seed=seed, verbose=0, device="cpu",
                       exploration_fraction=protocol["sb3_dqn_exploration_fraction"], **hyper)
    else:
        env = DummyVecEnv([partial(Slicer, bomb_ratio=bomb_ratio, speed=speed)
                           for _ in range(protocol["ppo_rollout_steps"] // hyper["n_steps"])])
        model = SB3PPO("MlpPolicy", env, seed=seed, verbose=0, device="cpu", **hyper)
    return model, env


def train_sb3(agent: str, seed: int, protocol: dict, bomb_ratio: float = 0.2,
              speed: float = 1.0, record=None):
    """Train one SB3 model; ``record(step, model)`` sees 0, every, ..., total_steps."""
    model, env = build_model(agent, seed, protocol, bomb_ratio, speed)
    total = protocol["total_steps"]
    callback = Checkpoints(protocol["checkpoint_every"], record) if record else None
    model.learn(total_timesteps=total, callback=callback)
    if model.num_timesteps != total:
        raise RuntimeError(f"{agent} consumed {model.num_timesteps} steps, expected {total}")
    if record is not None and total not in callback.steps:
        callback.steps.append(total)
        record(total, model)
    env.close()
    return model


def train_job(job: dict) -> dict:
    """Train one SB3 agent on one setting; evaluate the same evolving model at each checkpoint."""
    protocol, agent, seed = job["protocol"], job["agent"], job["seed"]
    settings, episodes = protocol["settings"], protocol["eval_episodes"]
    eval_seed = protocol["eval_seed_base"] + seed
    curve_seed = protocol["curve_eval_seed_base"] + seed
    checkpoints, model_ids = [], set()

    def record(step, model):
        model_ids.add(id(model))
        row = {"step": int(step), "param_sha256_16": curves.fingerprint(model.policy),
               "eval": {name: eval_sb3(model, seed=eval_seed, episodes=episodes, **settings[name])
                        for name in job["eval_settings"]}}
        if job["curve"]:
            row["consistency_eval"] = eval_sb3(model, seed=curve_seed, episodes=episodes)
        checkpoints.append(row)

    model = train_sb3(agent, seed, protocol, record=record, **settings[job["train_setting"]])
    if model_ids != {id(model)}:
        raise RuntimeError("checkpoints did not come from the returned training run")
    expected = list(range(0, protocol["total_steps"] + 1, protocol["checkpoint_every"]))
    if [c["step"] for c in checkpoints] != expected:
        raise RuntimeError(f"checkpoint steps {[c['step'] for c in checkpoints]} != {expected}")
    return {"agent": agent, "train_setting": job["train_setting"], "seed": seed,
            "eval_seed": eval_seed, "consistency_eval_seed": curve_seed if job["curve"] else None,
            "trained_steps": int(model.num_timesteps), "checkpoints": checkpoints}


def build_jobs(protocol: dict) -> list[dict]:
    train, everything = protocol["training_setting"], list(protocol["settings"])
    jobs = [{"protocol": protocol, "agent": agent, "seed": seed, "train_setting": train,
             "eval_settings": everything, "curve": True}
            for agent in protocol["agents"] for seed in protocol["training_seeds"]]
    jobs += [{"protocol": protocol, "agent": agent, "seed": seed, "train_setting": held,
              "eval_settings": [held], "curve": False}
             for held in protocol["held_out"] for agent in protocol["agents"]
             for seed in protocol["training_seeds"]]
    return jobs


# ---------------------------------------------------------------- summaries

def curve_rows(rows: list[dict]) -> list[dict]:
    """The default-setting checkpoints on the curve episodes, in analysis.curves row format."""
    return [{"agent": r["agent"], "seed": r["seed"], "eval_seed": r["consistency_eval_seed"],
             "checkpoints": [{"step": c["step"], "param_sha256_16": c["param_sha256_16"],
                              **c["consistency_eval"]} for c in r["checkpoints"]]}
            for r in rows if r["consistency_eval_seed"] is not None]


def implementation_gap(sb3_values, scratch_values) -> dict:
    gap = difficulty.gap_ci(sb3_values, scratch_values)
    low, high = gap["ci95"]
    gap["reading"] = ("SB3 higher" if low > 0 else "scratch higher" if high < 0
                      else "no measurable difference")
    return gap


def compare_with_scratch(rows: list[dict], protocol: dict, root: Path = ROOT) -> dict:
    """SB3 minus scratch twin at the final step, from the committed scratch evidence."""
    curve = json.loads((root / FROZEN_INPUTS["curve_evidence"]).read_text(encoding="utf-8"))
    scratch_rows = json.loads((root / FROZEN_INPUTS["difficulty_evidence"]).read_text(
        encoding="utf-8"))["rows"]
    train, final = protocol["training_setting"], protocol["total_steps"]
    out = {}
    for agent, twin in SCRATCH_TWIN.items():
        own_curve = sorted((r for r in curve_rows(rows) if r["agent"] == agent), key=lambda r: r["seed"])
        twin_curve = sorted((r for r in curve["rows"] if r["agent"] == twin), key=lambda r: r["seed"])
        if [r["seed"] for r in own_curve] != [r["seed"] for r in twin_curve]:
            raise ValueError("SB3 and scratch curves use different seeds")
        last = lambda group, metric: [r["checkpoints"][-1][metric] for r in group]  # noqa: E731
        if any(r["checkpoints"][-1]["step"] != final for r in own_curve + twin_curve):
            raise ValueError("last checkpoint is not the final step")
        entry = {"curve_episodes": {
            "score": implementation_gap(last(own_curve, "mean_score"), last(twin_curve, "mean_score")),
            "sb3_bomb_hits": sum(last(own_curve, "mean_bomb_hits")) / REQUIRED_SEEDS,
            "scratch_bomb_hits": sum(last(twin_curve, "mean_bomb_hits")) / REQUIRED_SEEDS}}
        for setting in protocol["settings"]:
            sb3 = difficulty.final_values(rows, agent, train, setting)
            scratch = difficulty.final_values(scratch_rows, twin, train, setting)
            entry[setting] = implementation_gap(sb3, scratch)
        out[agent] = {"scratch_twin": twin, "gaps": entry}
    return out


def historical_consistency(rows: list[dict], root: Path = ROOT) -> list[dict]:
    """Seed-by-seed comparison of SB3 checkpoints with the saved SB3 endpoint runs."""
    curve = {"rows": curve_rows(rows)}
    checks = []
    for name, step in (("endpoint_5120", 5120), ("endpoint_10240", 10240)):
        endpoint = json.loads((root / FROZEN_INPUTS[name]).read_text(encoding="utf-8"))
        for agent in SCRATCH_TWIN:
            checks.append({"endpoint_file": FROZEN_INPUTS[name],
                           **curves.endpoint_consistency(curve, endpoint, agent, step)})
    return checks


def anchors_for(protocol: dict) -> list[dict]:
    return [{"setting": name, "policy": policy, "seed": seed,
             "eval_seed": protocol["eval_seed_base"] + seed,
             **difficulty.evaluate_anchor(policy, protocol["eval_seed_base"] + seed,
                                          protocol["eval_episodes"], **kwargs)}
            for name, kwargs in protocol["settings"].items()
            for policy in ("random", "scripted") for seed in protocol["training_seeds"]]


# ---------------------------------------------------------------- run

def run(protocol_path: Path = PROTOCOL_PATH, out: Path = RUN_PATH, workers: int = 1,
        protocol_constant: dict = PROTOCOL, root: Path = ROOT,
        compare_with: Path | None = EVIDENCE_PATH, full_checks: bool = True) -> dict:
    if not 1 <= workers <= 4:
        raise ValueError("use between 1 and 4 worker processes")
    protocol = load_protocol(protocol_path, protocol_constant, root)
    started = datetime.now(UTC)
    jobs = build_jobs(protocol)
    if workers == 1:
        rows = [train_job(job) for job in jobs]
    else:
        context = multiprocessing.get_context("spawn")
        with ProcessPoolExecutor(max_workers=workers, mp_context=context) as pool:
            rows = list(pool.map(train_job, jobs))
    anchors = anchors_for(protocol)
    curve = curve_rows(rows)
    result = {
        "protocol": protocol["name"], "protocol_file": protocol_path.name,
        "protocol_sha256": sha256(protocol_path),
        "protocol_written_at_utc": protocol["written_at_utc"],
        "started_at_utc": started.isoformat(timespec="seconds"),
        "finished_at_utc": datetime.now(UTC).isoformat(timespec="seconds"),
        "workers": workers,
        "curve": {"threshold": 5.0, "rows": curve,
                  "summary": curves.summarize(curve, 5.0)},
        "difficulty_summary": difficulty.summarize(rows, anchors, protocol),
        "anchors": anchors, "rows": rows,
    }
    if full_checks:
        saved = json.loads((root / FROZEN_INPUTS["difficulty_evidence"]).read_text(encoding="utf-8"))
        result["anchors_match_difficulty_evidence"] = saved["anchors"] == json.loads(json.dumps(anchors))
        result["historical_endpoint_consistency"] = historical_consistency(rows, root)
        result["scratch_comparison"] = compare_with_scratch(rows, protocol, root)
    if compare_with is not None and compare_with.exists():
        saved = json.loads(compare_with.read_text(encoding="utf-8"))
        result["matches_committed_evidence"] = (saved["rows"] == json.loads(json.dumps(rows))
                                                and saved["anchors"] == json.loads(json.dumps(anchors)))
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    return result


def headline(result: dict) -> dict:
    curve = {agent: {"final": round(s["curve"][-1]["mean_score"]["mean"], 3),
                     "reached_5.0": f"{s['reached_threshold']}/{s['seeds']}"}
             for agent, s in result["curve"]["summary"].items()}
    table = {}
    for agent, data in result["difficulty_summary"]["agents"].items():
        for setting, entry in data["settings"].items():
            cell = {"score": round(entry["mean_score"]["mean"], 3)}
            if "transfer_gap" in entry:
                cell["gap"] = entry["transfer_gap"]["reading"]
            table[f"{agent}|{setting}"] = cell
    return {"curve": curve,
            "historical_consistency": {f"{c['agent']}@{c['step']}": c["identical"]
                                       for c in result.get("historical_endpoint_consistency", [])},
            "anchors_match_difficulty_evidence": result.get("anchors_match_difficulty_evidence"),
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
                          "frozen_inputs_sha256": body["frozen_inputs_sha256"]}, indent=2))
        return
    result = run(workers=args.workers)
    print(f"{RUN_PATH.relative_to(ROOT)} written")
    print(json.dumps(headline(result), indent=2))


if __name__ == "__main__":
    main()
