"""Run the predeclared reward-fault challenge against the frozen v2 oracle."""

from __future__ import annotations

import hashlib
import json
from datetime import UTC, datetime
from pathlib import Path
from statistics import mean, median

from envs.gridquest import GridQuest, Layout, default_layout
from playtest.bug_hunt import _scripted_action, hunt
from playtest.new_defect import held_out_layouts
from playtest.unseen import wilson

ROOT = Path(__file__).resolve().parents[1]
ORIGINAL_HUNT_ALLOWLIST = 'NEW_DEFECTS = ("coin_not_scored",)'
EXTENDED_HUNT_ALLOWLIST = 'NEW_DEFECTS = ("coin_not_scored", "goal_reward_missing")'


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_protocol(root: Path = ROOT) -> dict:
    protocol_path = root / "evidence" / "frozen_oracle_reward_protocol.json"
    protocol = json.loads(protocol_path.read_text())
    if protocol["name"] != "gridquest-frozen-oracle-reward-v1":
        raise ValueError("wrong reward challenge protocol")
    if protocol["defect"] != "goal_reward_missing":
        raise ValueError("wrong reward challenge defect")
    if protocol["policies"] != ["random", "scripted", "coverage", "q_count"]:
        raise ValueError("policy set changed")
    if protocol["hunt_seeds"] != [0, 1, 2, 3, 4] or protocol["max_steps"] != 1200:
        raise ValueError("hunt seeds or budget changed")
    for rel, frozen_hash in protocol["frozen_inputs_sha256"].items():
        path = root / rel
        if rel == "playtest/bug_hunt.py":
            current = path.read_text()
            if current.count(EXTENDED_HUNT_ALLOWLIST) != 1:
                raise RuntimeError("hunt allow-list differs from the declared extension")
            original = current.replace(EXTENDED_HUNT_ALLOWLIST, ORIGINAL_HUNT_ALLOWLIST)
            actual = hashlib.sha256(original.encode()).hexdigest()
        else:
            actual = sha256(path)
        if actual != frozen_hash:
            raise RuntimeError(f"frozen challenge input changed: {rel}")
    return protocol


def scripted_exposure(layout: Layout, seed: int, defect: str) -> dict:
    """Prove a goal reward was wrong without feeding this probe into the hunt."""
    env = GridQuest(layout=layout, bug=defect)
    env.reset(seed=seed)
    for step in range(1, env.max_steps + 1):
        _, reward, terminated, truncated, info = env.step(_scripted_action(env))
        if info["success"]:
            return {"seed": seed, "reached_goal": True, "step": step,
                    "observed_reward": reward, "fault_exposed": reward == 0.0}
        if terminated or truncated:
            break
    return {"seed": seed, "reached_goal": False, "step": None,
            "observed_reward": None, "fault_exposed": False}


def summarize(rows: list[dict], protocol: dict) -> dict:
    summary = {}
    for map_set, n in (("default", 5), ("held_out", 40)):
        cells = []
        for policy in protocol["policies"]:
            defect_rows = [row for row in rows if row["map_set"] == map_set
                           and row["policy"] == policy and row["defect"] is not None]
            clean_rows = [row for row in rows if row["map_set"] == map_set
                          and row["policy"] == policy and row["defect"] is None]
            if len(defect_rows) != n or len(clean_rows) != n:
                raise ValueError(f"incomplete {map_set}/{policy} rows")
            found = [row["time_to_discovery"] for row in defect_rows
                     if row["time_to_discovery"] is not None]
            kinds = {}
            for row in defect_rows:
                if row["detected"]:
                    kinds[row["detected"]] = kinds.get(row["detected"], 0) + 1
            alarms = sum(row["detected"] is not None for row in clean_rows)
            cells.append({
                "policy": policy, "detected": len(found), "n": n,
                "wilson_95": wilson(len(found), n), "censored": n - len(found),
                "mean_step_detected_only": round(mean(found), 1) if found else None,
                "median_step_detected_only": median(found) if found else None,
                "violation_kinds": kinds, "clean_false_alarms": alarms,
            })
        summary[map_set] = cells
    return summary


def run(root: Path = ROOT) -> dict:
    protocol = load_protocol(root)
    maps = {"default": [default_layout()], "held_out": held_out_layouts(root)}
    if len(maps["held_out"]) != 8:
        raise ValueError("expected eight frozen held-out maps")
    started = datetime.now(UTC)
    rows = []
    exposures = []
    for map_set, layouts in maps.items():
        for index, layout in enumerate(layouts):
            for seed in protocol["hunt_seeds"]:
                exposures.append({"map_set": map_set, "map": index,
                                  **scripted_exposure(layout, seed, protocol["defect"])})
            for defect in (None, protocol["defect"]):
                for policy in protocol["policies"]:
                    for seed in protocol["hunt_seeds"]:
                        outcome = hunt(defect, policy, seed, protocol["max_steps"],
                                       layout=layout, oracle="v2")
                        rows.append({
                            "map_set": map_set, "map": index, "defect": defect,
                            "policy": policy, "seed": seed,
                            "time_to_discovery": outcome["time_to_discovery"],
                            "detected": outcome["detected"],
                            "position_coverage": round(outcome["position_coverage"], 4),
                        })
    result = {
        "protocol": protocol["name"],
        "protocol_sha256": sha256(root / "evidence" / "frozen_oracle_reward_protocol.json"),
        "started_at_utc": started.isoformat(timespec="seconds"),
        "finished_at_utc": datetime.now(UTC).isoformat(timespec="seconds"),
        "summary": summarize(rows, protocol),
        "scripted_exposure": {"fault_exposed": sum(row["fault_exposed"] for row in exposures),
                              "n": len(exposures), "rows": exposures},
        "rows": rows,
    }
    evidence_path = root / "evidence" / "frozen_oracle_reward_results.json"
    if evidence_path.exists():
        saved = json.loads(evidence_path.read_text())
        result["matches_committed_evidence"] = (
            result["rows"] == saved["rows"]
            and result["scripted_exposure"] == saved["scripted_exposure"]
        )
    output = root / "runs" / "frozen_oracle_reward.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2) + "\n")
    return result


if __name__ == "__main__":
    result = run()
    print(json.dumps({"summary": result["summary"],
                      "scripted_exposure": {key: result["scripted_exposure"][key]
                                            for key in ("fault_exposed", "n")},
                      "matches_committed_evidence": result.get("matches_committed_evidence")},
                     indent=2))
