"""Run the frozen, budget-matched 40,960-step Slicer bomb comparison."""

from __future__ import annotations

import hashlib
import json
import multiprocessing
from concurrent.futures import ProcessPoolExecutor
from datetime import UTC, datetime
from pathlib import Path

from analysis import difficulty, sb3_reference
from analysis.aggregate import mean_ci

ROOT = Path(__file__).resolve().parents[1]
PROTOCOL = ROOT / "evidence" / "slicer_bomb_matched_40960_protocol.json"
SAVED = ROOT / "evidence" / "slicer_bomb_matched_40960_results.json"
OUTPUT = ROOT / "runs" / "slicer_bomb_matched_40960.json"
METRICS = ("mean_score", "mean_bomb_hits", "mean_return")
FROZEN_PROTOCOL_SHA256 = "9c4902a67fd4b4e3594bf767f03feb950d62dc93468841096d5690c4af50f8eb"


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_protocol(root: Path = ROOT) -> dict:
    path = root / PROTOCOL.relative_to(ROOT)
    if digest(path) != FROZEN_PROTOCOL_SHA256:
        raise RuntimeError("budget-matched protocol changed after its separate commit")
    protocol = json.loads(path.read_text())
    if (protocol["name"] != "slicer-bomb-matched-40960-v1"
            or protocol["agents"] != ["double_dqn", "scratch_ppo", "sb3_dqn", "sb3_ppo"]
            or protocol["seeds"] != [0, 1, 2, 3, 4]
            or protocol["checkpoints"] != [20480, 40960]
            or protocol["training_settings"] != list(protocol["settings"])):
        raise ValueError("budget-matched protocol structure changed")
    for rel, expected in protocol["frozen_inputs_sha256"].items():
        if digest(root / rel) != expected:
            raise RuntimeError(f"frozen budget-matched input changed: {rel}")
    return protocol


def _run_job(item: tuple[str, dict]) -> dict:
    agent, job = item
    if agent.startswith("sb3_"):
        return sb3_reference.train_job(job)
    return difficulty.train_job(job)


def build_jobs(protocol: dict) -> list[tuple[str, dict]]:
    job_protocol = {
        "settings": protocol["settings"], "total_steps": protocol["checkpoints"][-1],
        "checkpoint_every": protocol["checkpoints"][0],
        "eval_episodes": protocol["eval_episodes"],
        "eval_seed_base": protocol["eval_seed_base"],
        "consistency_eval_seed_base": 1000, "curve_eval_seed_base": 1000,
        "dqn_epsilon_decay_steps": protocol["dqn_epsilon_decay_steps"],
        "sb3_dqn_exploration_fraction": protocol["sb3_dqn_exploration_fraction"],
        "ppo_rollout_steps": protocol["ppo_rollout_steps"],
        "hyperparameters": protocol["sb3_hyperparameters"],
    }
    jobs = []
    for agent in protocol["agents"]:
        for setting in protocol["training_settings"]:
            if agent == "double_dqn" and setting != "train":
                continue  # Frozen 31 October rows are reused, not retrained.
            for seed in protocol["seeds"]:
                job = {"protocol": job_protocol, "agent": agent, "seed": seed,
                       "train_setting": setting,
                       "eval_settings": list(protocol["settings"]) if setting == "train"
                       else [setting],
                       "consistency": False, "curve": False}
                jobs.append((agent, job))
    if len(jobs) != 65:
        raise RuntimeError("expected 65 new training jobs")
    return jobs


def compare_prior(rows: list[dict], root: Path = ROOT) -> dict:
    sources = [json.loads((root / "evidence" / name).read_text()) for name in
               ("slicer_difficulty_20480.json", "sb3_reference_20480.json")]
    prior = {(row["agent"], row["train_setting"], row["seed"]): row
             for source in sources for row in source["rows"]}
    mismatches = []
    for row in rows:
        key = (row["agent"], row["train_setting"], row["seed"])
        old = prior[key]["checkpoints"][-1]
        current = next(c for c in row["checkpoints"] if c["step"] == 20480)
        if (current["param_sha256_16"] != old["param_sha256_16"]
                or any(current["eval"][setting][metric] != old["eval"][setting][metric]
                       for setting in current["eval"] for metric in METRICS)):
            mismatches.append({"agent": key[0], "setting": key[1], "seed": key[2]})
    return {"compared": len(rows), "identical": len(rows) - len(mismatches),
            "mismatches": mismatches}


def summarize(rows: list[dict], protocol: dict) -> dict:
    cells = {}
    for agent in protocol["agents"]:
        cells[agent] = {}
        for setting in protocol["training_settings"][1:]:
            groups = {}
            for trained_on in ("train", setting):
                group = sorted((r for r in rows if r["agent"] == agent
                                and r["train_setting"] == trained_on),
                               key=lambda r: r["seed"])
                if [r["seed"] for r in group] != protocol["seeds"]:
                    raise ValueError(f"missing or repeated seed: {agent}/{trained_on}")
                groups[trained_on] = {
                    metric: mean_ci([next(c for c in row["checkpoints"] if c["step"] == 40960)
                                     ["eval"][setting][metric] for row in group])
                    for metric in METRICS
                }
            gaps = {metric: difficulty.gap_ci(
                groups[setting][metric]["seed_values"],
                groups["train"][metric]["seed_values"])
                for metric in ("mean_score", "mean_return")}
            cells[agent][setting] = {"default_trained": groups["train"],
                                     "in_setting": groups[setting],
                                     "in_setting_minus_default": gaps}
    return cells


def run(root: Path = ROOT, workers: int = 4) -> dict:
    if not 1 <= workers <= 4:
        raise ValueError("use 1–4 worker processes")
    protocol = load_protocol(root)
    prior_dqn = json.loads((root / "evidence" / "slicer_bomb_dqn_40960_results.json").read_text())
    reused = prior_dqn["rows"]
    if len(reused) != 15 or prior_dqn["prior_reproduction"] != {
            "compared": 15, "identical": 15, "mismatches": []}:
        raise RuntimeError("earlier longer DQN rows are incomplete")
    started = datetime.now(UTC)
    jobs = build_jobs(protocol)
    if workers == 1:
        new_rows = [_run_job(item) for item in jobs]
    else:
        context = multiprocessing.get_context("spawn")
        with ProcessPoolExecutor(max_workers=workers, mp_context=context) as pool:
            new_rows = list(pool.map(_run_job, jobs))
    rows = sorted([*new_rows, *reused],
                  key=lambda row: (protocol["agents"].index(row["agent"]),
                                   protocol["training_settings"].index(row["train_setting"]),
                                   row["seed"]))
    reproduction = compare_prior(rows, root)
    result = {
        "protocol": protocol["name"], "protocol_sha256": digest(root / PROTOCOL.relative_to(ROOT)),
        "started_at_utc": started.isoformat(timespec="seconds"),
        "finished_at_utc": datetime.now(UTC).isoformat(timespec="seconds"),
        "workers": workers, "new_training_jobs": len(jobs), "reused_jobs": len(reused),
        "prior_reproduction": reproduction,
        "summary": summarize(rows, protocol) if not reproduction["mismatches"] else None,
        "rows": rows,
    }
    saved_path = root / SAVED.relative_to(ROOT)
    if saved_path.exists():
        saved = json.loads(saved_path.read_text())
        result["matches_committed_evidence"] = (
            result["rows"] == saved["rows"] and result["summary"] == saved["summary"])
    output = root / OUTPUT.relative_to(ROOT)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2) + "\n")
    return result


if __name__ == "__main__":
    result = run()
    print(json.dumps({"prior_reproduction": result["prior_reproduction"],
                      "new_training_jobs": result["new_training_jobs"],
                      "matches_committed_evidence": result.get("matches_committed_evidence"),
                      "summary": result["summary"]}, indent=2))
