"""Run the frozen 40,960-step scratch DQN bomb-setting reference check."""

from __future__ import annotations

import hashlib
import json
import multiprocessing
from concurrent.futures import ProcessPoolExecutor
from datetime import UTC, datetime
from pathlib import Path

from analysis.aggregate import mean_ci
from analysis.difficulty import train_job

ROOT = Path(__file__).resolve().parents[1]
PROTOCOL = ROOT / "evidence" / "slicer_bomb_dqn_40960_protocol.json"
PRIOR = ROOT / "evidence" / "slicer_difficulty_20480.json"
SAVED = ROOT / "evidence" / "slicer_bomb_dqn_40960_results.json"
OUTPUT = ROOT / "runs" / "slicer_bomb_dqn_40960.json"
METRICS = ("mean_bomb_hits", "mean_return", "mean_score")
FROZEN_PROTOCOL_SHA256 = "db8ce4b8f6957e9ac0236a84e81d3839db8aca56117278508ed4f263e4c3d49f"


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_protocol(root: Path = ROOT) -> dict:
    protocol_path = root / PROTOCOL.relative_to(ROOT)
    if digest(protocol_path) != FROZEN_PROTOCOL_SHA256:
        raise RuntimeError("longer DQN protocol changed after its separate commit")
    protocol = json.loads(protocol_path.read_text())
    if (protocol["name"] != "slicer-bomb-dqn-40960-v1"
            or protocol["agent"] != "double_dqn"
            or protocol["seeds"] != [0, 1, 2, 3, 4]
            or protocol["steps"] != [20480, 40960]
            or protocol["dqn_epsilon_decay_steps"] != 3584
            or protocol["eval_seed_base"] != 7000
            or protocol["eval_episodes"] != 30
            or list(protocol["settings"]) != ["bombs_0.4", "bombs_0.6",
                                                "bombs_0.4_fast_1.3"]):
        raise ValueError("longer DQN protocol fields changed")
    for rel, expected in protocol["frozen_inputs_sha256"].items():
        if digest(root / rel) != expected:
            raise RuntimeError(f"frozen longer DQN input changed: {rel}")
    prior = json.loads((root / "evidence" / "slicer_difficulty_protocol.json").read_text())
    for setting, knobs in protocol["settings"].items():
        if prior["settings"][setting] != knobs:
            raise RuntimeError(f"prior setting differs: {setting}")
    return protocol


def compare_prior(rows: list[dict], prior: dict) -> dict:
    old = {(row["train_setting"], row["seed"]): row
           for row in prior["rows"] if row["agent"] == "double_dqn"
           and row["train_setting"] in {r["train_setting"] for r in rows}}
    mismatches = []
    for row in rows:
        prior_row = old[(row["train_setting"], row["seed"])]
        current = next(c for c in row["checkpoints"] if c["step"] == 20480)
        previous = prior_row["checkpoints"][-1]
        if (current["param_sha256_16"] != previous["param_sha256_16"]
                or any(current["eval"][row["train_setting"]][metric]
                       != previous["eval"][row["train_setting"]][metric]
                       for metric in METRICS)):
            mismatches.append({"setting": row["train_setting"], "seed": row["seed"]})
    return {"compared": len(rows), "identical": len(rows) - len(mismatches),
            "mismatches": mismatches}


def summary(rows: list[dict], protocol: dict) -> dict:
    cells = {}
    for setting in protocol["settings"]:
        group = sorted((row for row in rows if row["train_setting"] == setting),
                       key=lambda row: row["seed"])
        if [row["seed"] for row in group] != protocol["seeds"]:
            raise ValueError(f"missing or repeated seed for {setting}")
        cells[setting] = {}
        for step in protocol["steps"]:
            cells[setting][str(step)] = {
                metric: mean_ci([next(c for c in row["checkpoints"] if c["step"] == step)
                                 ["eval"][setting][metric] for row in group])
                for metric in METRICS
            }
    return cells


def run(workers: int = 4, root: Path = ROOT) -> dict:
    if not 1 <= workers <= 4:
        raise ValueError("use 1–4 worker processes")
    protocol = load_protocol(root)
    prior = json.loads((root / PRIOR.relative_to(ROOT)).read_text())
    job_protocol = {"settings": protocol["settings"],
                    "total_steps": protocol["steps"][-1],
                    "checkpoint_every": protocol["steps"][0],
                    "dqn_epsilon_decay_steps": protocol["dqn_epsilon_decay_steps"],
                    "eval_episodes": protocol["eval_episodes"],
                    "eval_seed_base": protocol["eval_seed_base"],
                    "consistency_eval_seed_base": 1000}
    jobs = [{"protocol": job_protocol, "agent": protocol["agent"],
             "seed": seed, "train_setting": setting, "eval_settings": [setting],
             "consistency": False}
            for setting in protocol["settings"] for seed in protocol["seeds"]]
    started = datetime.now(UTC)
    if workers == 1:
        rows = [train_job(job) for job in jobs]
    else:
        context = multiprocessing.get_context("spawn")
        with ProcessPoolExecutor(max_workers=workers, mp_context=context) as pool:
            rows = list(pool.map(train_job, jobs))
    reproduction = compare_prior(rows, prior)
    result = {"protocol": protocol["name"],
              "protocol_sha256": digest(root / PROTOCOL.relative_to(ROOT)),
              "started_at_utc": started.isoformat(timespec="seconds"),
              "finished_at_utc": datetime.now(UTC).isoformat(timespec="seconds"),
              "workers": workers, "prior_reproduction": reproduction,
              "summary": summary(rows, protocol) if not reproduction["mismatches"] else None,
              "rows": rows}
    saved_path = root / SAVED.relative_to(ROOT)
    if saved_path.exists():
        saved = json.loads(saved_path.read_text())
        result["matches_committed_evidence"] = (result["rows"] == saved["rows"])
    output = root / OUTPUT.relative_to(ROOT)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2) + "\n")
    return result


if __name__ == "__main__":
    result = run()
    print(json.dumps({"prior_reproduction": result["prior_reproduction"],
                      "summary": result["summary"],
                      "matches_committed_evidence": result.get("matches_committed_evidence")},
                     indent=2))
