"""Score the separately frozen 2 November Slicer settings without tuning."""

from __future__ import annotations

import argparse
import hashlib
import json
import multiprocessing
from concurrent.futures import ProcessPoolExecutor
from datetime import UTC, datetime
from pathlib import Path

from analysis import difficulty, sb3_reference
from analysis.aggregate import mean_ci

ROOT = Path(__file__).resolve().parents[1]
PROTOCOL = ROOT / "evidence/slicer_fresh_transfer_40960_protocol.json"
PRIOR = ROOT / "evidence/slicer_bomb_matched_40960_results.json"
SAVED = ROOT / "evidence/slicer_fresh_transfer_40960_results.json"
OUTPUT = ROOT / "runs/slicer_fresh_transfer_40960.json"
PROTOCOL_SHA256 = "85a92a30e9ebae2289fba293f3234d120dda90ff33ee1a893bfe8ddab34e0b5f"
METRICS = ("mean_score", "mean_bomb_hits", "mean_return")


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_protocol(root: Path = ROOT) -> dict:
    path = root / PROTOCOL.relative_to(ROOT)
    if digest(path) != PROTOCOL_SHA256:
        raise RuntimeError("fresh transfer protocol differs from its pre-training commit")
    p = json.loads(path.read_text())
    if (p["name"] != "slicer-fresh-transfer-40960-v1"
            or p["agents"] != ["double_dqn", "scratch_ppo", "sb3_dqn", "sb3_ppo"]
            or p["training_seeds"] != list(range(7))
            or p["checkpoints"] != [0, 20480, 40960]
            or p["eval_seed_base"] != 9100
            or p["held_out"] != list(p["settings"])[1:]
            or p["training_setting"] != "train"
            or p["sb3_dqn_exploration_fraction"] != p["dqn_epsilon_decay_steps"] / 40960):
        raise ValueError("fresh transfer protocol structure changed")
    for rel, expected in p["frozen_inputs_sha256"].items():
        if digest(root / rel) != expected:
            raise RuntimeError(f"frozen input changed: {rel}")
    old = json.loads((root / "evidence/slicer_difficulty_protocol.json").read_text())
    old_settings = {tuple(sorted(value.items())) for value in old["settings"].values()}
    for setting in p["held_out"]:
        if tuple(sorted(p["settings"][setting].items())) in old_settings:
            raise RuntimeError(f"new setting was in the earlier difficulty protocol: {setting}")
    return p


def build_jobs(p: dict) -> list[tuple[str, dict]]:
    job_protocol = {
        "settings": p["settings"], "total_steps": p["checkpoints"][-1],
        "checkpoint_every": p["checkpoints"][1],
        "eval_episodes": p["eval_episodes"], "eval_seed_base": p["eval_seed_base"],
        "consistency_eval_seed_base": 1000, "curve_eval_seed_base": 1000,
        "dqn_epsilon_decay_steps": p["dqn_epsilon_decay_steps"],
        "sb3_dqn_exploration_fraction": p["sb3_dqn_exploration_fraction"],
        "ppo_rollout_steps": p["ppo_rollout_steps"],
        "hyperparameters": p["sb3_hyperparameters"],
    }
    jobs = []
    for agent in p["agents"]:
        for trained_on in p["settings"]:
            for seed in p["training_seeds"]:
                job = {"protocol": job_protocol, "agent": agent, "seed": seed,
                       "train_setting": trained_on,
                       "eval_settings": p["held_out"] if trained_on == "train" else [trained_on],
                       "consistency": False, "curve": False}
                jobs.append((agent, job))
    if len(jobs) != 84:
        raise RuntimeError(f"expected 84 jobs, got {len(jobs)}")
    return jobs


def _run_job(item: tuple[str, dict]) -> dict:
    agent, job = item
    return sb3_reference.train_job(job) if agent.startswith("sb3_") else difficulty.train_job(job)


def check_prior_default(rows: list[dict], root: Path = ROOT) -> dict:
    old = json.loads((root / PRIOR.relative_to(ROOT)).read_text())
    prior = {(r["agent"], r["seed"]): r for r in old["rows"] if r["train_setting"] == "train"}
    current = {(r["agent"], r["seed"]): r for r in rows if r["train_setting"] == "train"
               and r["seed"] < 5}
    expected = {(agent, seed) for agent in old["summary"] for seed in range(5)}
    if set(current) != expected or not expected <= set(prior):
        raise RuntimeError("default-trained reproduction rows are incomplete")
    mismatches = []
    for key in sorted(expected):
        for step in (20480, 40960):
            before = next(c for c in prior[key]["checkpoints"] if c["step"] == step)
            now = next(c for c in current[key]["checkpoints"] if c["step"] == step)
            if now["param_sha256_16"] != before["param_sha256_16"]:
                mismatches.append({"agent": key[0], "seed": key[1], "step": step})
    return {"compared": 40, "identical": 40 - len(mismatches), "mismatches": mismatches}


def _values(rows: list[dict], agent: str, trained_on: str, setting: str,
            metric: str, seeds: list[int]) -> list[float]:
    group = sorted((r for r in rows if r["agent"] == agent
                    and r["train_setting"] == trained_on), key=lambda r: r["seed"])
    if [r["seed"] for r in group] != seeds:
        raise ValueError(f"missing or repeated seed: {agent}/{trained_on}")
    if any([c["step"] for c in r["checkpoints"]] != [0, 20480, 40960] for r in group):
        raise ValueError(f"checkpoint steps differ: {agent}/{trained_on}")
    return [r["checkpoints"][-1]["eval"][setting][metric] for r in group]


def summarize(rows: list[dict], anchors: list[dict], p: dict) -> dict:
    cells = {}
    for agent in p["agents"]:
        cells[agent] = {}
        for setting in p["held_out"]:
            groups = {}
            for trained_on in ("train", setting):
                groups[trained_on] = {metric: mean_ci(_values(
                    rows, agent, trained_on, setting, metric, p["training_seeds"]))
                    for metric in METRICS}
            gaps = {metric: difficulty.gap_ci(
                groups[setting][metric]["seed_values"],
                groups["train"][metric]["seed_values"])
                for metric in ("mean_return", "mean_score")}
            cells[agent][setting] = {"default_trained": groups["train"],
                                     "in_setting": groups[setting],
                                     "in_setting_minus_default": gaps}
    if len(anchors) != len(p["held_out"]) * 2 * len(p["training_seeds"]):
        raise ValueError("anchor rows are incomplete")
    return {"primary": {"agent": p["primary_comparison"]["agent"],
                        "setting": p["primary_comparison"]["setting"],
                        "metric": p["primary_comparison"]["metric"],
                        "gap": cells[p["primary_comparison"]["agent"]]
                                    [p["primary_comparison"]["setting"]]
                                    ["in_setting_minus_default"]["mean_return"]},
            "cells": cells,
            "anchors": {setting: {policy: {metric: mean_ci([
                a[metric] for a in anchors if a["setting"] == setting and a["policy"] == policy])
                for metric in METRICS} for policy in ("random", "scripted")}
                for setting in p["held_out"]}}


def run(root: Path = ROOT, workers: int = 4) -> dict:
    if not 1 <= workers <= 4:
        raise ValueError("use 1–4 workers")
    p = load_protocol(root)
    started = datetime.now(UTC)
    jobs = build_jobs(p)
    if workers == 1:
        rows = [_run_job(job) for job in jobs]
    else:
        context = multiprocessing.get_context("spawn")
        with ProcessPoolExecutor(max_workers=workers, mp_context=context) as pool:
            rows = list(pool.map(_run_job, jobs))
    rows.sort(key=lambda r: (p["agents"].index(r["agent"]),
                             list(p["settings"]).index(r["train_setting"]), r["seed"]))
    reproduction = check_prior_default(rows, root)
    anchors = [{"setting": setting, "policy": policy, "seed": seed,
                **difficulty.evaluate_anchor(policy, p["eval_seed_base"] + seed,
                                             p["eval_episodes"], **p["settings"][setting])}
               for setting in p["held_out"] for policy in ("random", "scripted")
               for seed in p["training_seeds"]]
    result = {"protocol": p["name"], "protocol_sha256": digest(root / PROTOCOL.relative_to(ROOT)),
              "started_at_utc": started.isoformat(timespec="seconds"),
              "finished_at_utc": datetime.now(UTC).isoformat(timespec="seconds"),
              "workers": workers, "training_jobs": len(jobs), "prior_default_reproduction": reproduction,
              "summary": summarize(rows, anchors, p) if not reproduction["mismatches"] else None,
              "rows": rows, "anchor_rows": anchors}
    saved_path = root / SAVED.relative_to(ROOT)
    if saved_path.exists():
        saved = json.loads(saved_path.read_text())
        result["matches_committed_evidence"] = (result["rows"] == saved["rows"]
                                                and result["anchor_rows"] == saved["anchor_rows"]
                                                and result["summary"] == saved["summary"])
    output = root / OUTPUT.relative_to(ROOT)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2) + "\n")
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--workers", type=int, default=4)
    args = parser.parse_args()
    result = run(workers=args.workers)
    print(json.dumps({"training_jobs": result["training_jobs"],
                      "prior_default_reproduction": result["prior_default_reproduction"],
                      "primary": result["summary"]["primary"] if result["summary"] else None,
                      "matches_committed_evidence": result.get("matches_committed_evidence")}, indent=2))
