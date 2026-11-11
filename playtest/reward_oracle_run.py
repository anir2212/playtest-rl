"""Run the separately committed gridquest-reward-oracle-v1 protocol for contract v3.

Refuses to start if the protocol file or a frozen input differs from its committed hash. Writes
only ``runs/reward_oracle.json``; the committed copy of the first scored run is
``evidence/reward_oracle_results.json``, and later runs report whether they match it.
"""
from __future__ import annotations

import hashlib
import json
import subprocess
from datetime import UTC, datetime
from pathlib import Path
from statistics import mean, median

from envs.gridquest import default_layout
from playtest.bug_hunt import hunt
from playtest.new_defect import held_out_layouts
from playtest.reward_oracle import hunt_v3
from playtest.unseen import wilson

ROOT = Path(__file__).resolve().parents[1]
PROTOCOL_REL = "evidence/reward_oracle_protocol.json"
EVIDENCE_REL = "evidence/reward_oracle_results.json"
OUTPUT_REL = "runs/reward_oracle.json"
PROTOCOL_SHA256 = "7e4478037a563edb72d97bbef39b08b5b72bc02a92f1365b1230f1082f692adf"
FROZEN_V2_RESULTS = "evidence/frozen_oracle_reward_results.json"
NEW_DEFECT_RESULTS = "evidence/new_defect_results.json"
ROW_FIELDS = ("time_to_discovery", "detected", "position_coverage")
MAP_SETS = {"default": 5, "held_out": 40}


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_protocol(root: Path = ROOT) -> dict:
    path = root / PROTOCOL_REL
    if digest(path) != PROTOCOL_SHA256:
        raise RuntimeError("reward oracle protocol differs from its committed hash")
    protocol = json.loads(path.read_text())
    if (protocol["name"] != "gridquest-reward-oracle-v1"
            or protocol["policies"] != ["random", "scripted", "coverage", "q_count"]
            or protocol["hunt_seeds"] != [0, 1, 2, 3, 4] or protocol["max_steps"] != 1200
            or protocol["defects"]["reward_fault"] != "goal_reward_missing"):
        raise ValueError("reward oracle protocol structure changed")
    for rel, frozen_hash in protocol["frozen_inputs_sha256"].items():
        if digest(root / rel) != frozen_hash:
            raise RuntimeError(f"frozen input changed: {rel}")
    return protocol


def conditions(protocol: dict) -> list[str | None]:
    return [None, *protocol["defects"]["earlier_catalogue"], protocol["defects"]["reward_fault"]]


def _key(row: dict) -> tuple:
    return row["map_set"], row["map"], row["policy"], row["seed"]


def _select(rows: list[dict], **fields) -> list[dict]:
    return [row for row in rows if all(row[name] == value for name, value in fields.items())]


def _same(rows: list[dict], reference: list[dict]) -> dict:
    lookup = {_key(row): row for row in reference}
    identical = sum(_key(row) in lookup and all(row[f] == lookup[_key(row)][f] for f in ROW_FIELDS)
                    for row in rows)
    return {"compared": len(rows), "reference": len(reference), "identical": identical,
            "all_identical": len(rows) == len(reference) == identical}


def committed_references(root: Path = ROOT) -> dict[str, dict]:
    """Earlier v2 rows that the control hunts must reproduce, keyed by the check's name."""
    frozen = json.loads((root / FROZEN_V2_RESULTS).read_text())["rows"]
    coin = _select(json.loads((root / NEW_DEFECT_RESULTS).read_text())["rows"], oracle="v2")
    return {"frozen_challenge": {"goal_reward_missing": _select(frozen, defect="goal_reward_missing"),
                                 "clean": _select(frozen, defect=None)},
            "new_defect": {"coin_not_scored": _select(coin, defect="coin_not_scored"),
                           "clean": _select(coin, defect=None)}}


def controls(rows: list[dict], references: dict[str, dict]) -> dict:
    out = {}
    for source, groups in references.items():
        for name, reference in groups.items():
            defect = None if name == "clean" else name
            out[f"{source}|{name}"] = _same(_select(rows, oracle="v2", defect=defect), reference)
    return out


def compare_v3_with_v2(rows: list[dict], protocol: dict) -> dict:
    """Paired v3-versus-v2 hunts on each earlier defect; every difference is listed."""
    out = {}
    for defect in protocol["defects"]["earlier_catalogue"]:
        v2 = {_key(row): row for row in _select(rows, oracle="v2", defect=defect)}
        pairs = _select(rows, oracle="v3", defect=defect)
        differences, lost = [], 0
        for row in pairs:
            base = v2[_key(row)]
            if any(row[field] != base[field] for field in ROW_FIELDS):
                differences.append({"key": list(_key(row)),
                                    "v2": [base["time_to_discovery"], base["detected"]],
                                    "v3": [row["time_to_discovery"], row["detected"]]})
            if base["time_to_discovery"] is not None and (
                    row["time_to_discovery"] is None
                    or row["time_to_discovery"] > base["time_to_discovery"]):
                lost += 1
        out[defect] = {"pairs": len(pairs), "differences": differences, "lost_or_later": lost}
    return out


def reward_exposure(rows: list[dict], protocol: dict) -> dict:
    """Separate 'never reached the fault' from 'saw it and stayed silent' under v3."""
    fault = _select(rows, oracle="v3", defect=protocol["defects"]["reward_fault"])
    exposed = [row for row in fault if row["first_success_step"] is not None]
    at_first = [row for row in exposed if row["detected"] == "reward_mismatch"
                and row["time_to_discovery"] == row["first_success_step"]]
    return {"n": len(fault), "exposed": len(exposed), "detected_at_first_exposure": len(at_first),
            "exposed_but_missed": sum(row["detected"] is None for row in exposed),
            "detected_without_exposure": sum(row["first_success_step"] is None
                                             and row["detected"] is not None for row in fault),
            "scripted_detected": sum(row["detected"] is not None for row in fault
                                     if row["policy"] == "scripted"),
            "scripted_n": sum(row["policy"] == "scripted" for row in fault)}


def acceptance(rows: list[dict], protocol: dict, control: dict, clean_loop: dict) -> dict:
    clean = _select(rows, oracle="v3", defect=None)
    alarms = sum(row["detected"] is not None for row in clean)
    exposure = reward_exposure(rows, protocol)
    paired = compare_v3_with_v2(rows, protocol)
    v2_fault = _select(rows, oracle="v2", defect=protocol["defects"]["reward_fault"])
    a = len(clean) == 180 and alarms == 0
    b = (exposure["scripted_n"] == exposure["scripted_detected"] == 45
         and exposure["detected_at_first_exposure"] == exposure["exposed"]
         and exposure["detected_without_exposure"] == 0)
    c = all(item["pairs"] == 180 and item["lost_or_later"] == 0 for item in paired.values())
    d = (len(v2_fault) == 180 and not any(row["detected"] for row in v2_fault)
         and all(item["all_identical"] for item in control.values())
         and clean_loop["all_identical"])
    return {"A_no_false_alarms": a, "B_reward_fault": b, "C_no_regression": c, "D_controls": d,
            "overall": "PASS" if a and b and c and d else "FAIL"}


def summarize(rows: list[dict], protocol: dict) -> dict:
    cells, totals, clean_reach = [], {}, []
    for oracle in protocol["oracles"]:
        for condition in conditions(protocol):
            group = _select(rows, oracle=oracle, defect=condition)
            detected = sum(row["detected"] is not None for row in group)
            label = condition or "clean"
            totals[f"{oracle}|{label}"] = {"detected": detected, "n": len(group),
                                           "wilson_95": wilson(detected, len(group)),
                                           "censored": len(group) - detected}
            for map_set, n in MAP_SETS.items():
                for policy in protocol["policies"]:
                    cell = _select(group, map_set=map_set, policy=policy)
                    if len(cell) != n:
                        raise ValueError(f"{oracle}/{label}/{map_set}/{policy}: expected {n} hunts")
                    found = [r["time_to_discovery"] for r in cell if r["detected"] is not None]
                    kinds: dict[str, int] = {}
                    for row in cell:
                        if row["detected"]:
                            kinds[row["detected"]] = kinds.get(row["detected"], 0) + 1
                    cells.append({
                        "oracle": oracle, "condition": label, "map_set": map_set, "policy": policy,
                        "detected": len(found), "n": n, "wilson_95": wilson(len(found), n),
                        "censored": n - len(found),
                        "mean_step_detected_only": round(mean(found), 1) if found else None,
                        "median_step_detected_only": median(found) if found else None,
                        "violation_kinds": kinds})
                    if oracle == "v3" and condition is None:
                        clean_reach.append({
                            "map_set": map_set, "policy": policy, "n": n,
                            "reached_goal_with_key": sum(r["first_success_step"] is not None
                                                         for r in cell)})
    return {"totals": totals, "cells": cells, "clean_v3_reached_goal": clean_reach,
            "reward_fault_exposure_v3": reward_exposure(rows, protocol),
            "paired_v3_vs_v2": compare_v3_with_v2(rows, protocol)}


def run_hunts(protocol: dict, maps: dict) -> tuple[list[dict], dict]:
    rows, clean_compared, clean_identical = [], 0, 0
    for map_set, layouts in maps.items():
        for index, layout in enumerate(layouts):
            for condition in conditions(protocol):
                for policy in protocol["policies"]:
                    for seed in protocol["hunt_seeds"]:
                        base = {"map_set": map_set, "map": index, "defect": condition,
                                "policy": policy, "seed": seed}
                        v2 = hunt(condition, policy, seed, protocol["max_steps"],
                                  layout=layout, oracle="v2")
                        v3 = hunt_v3(condition, policy, seed, protocol["max_steps"], layout=layout)
                        for oracle, outcome in (("v2", v2), ("v3", v3)):
                            rows.append({**base, "oracle": oracle,
                                         "time_to_discovery": outcome["time_to_discovery"],
                                         "detected": outcome["detected"],
                                         "position_coverage": round(outcome["position_coverage"], 4),
                                         "first_success_step": outcome.get("first_success_step")})
                        if condition is None:
                            clean_compared += 1
                            clean_identical += (v2["position_visits"] == v3["position_visits"]
                                                and v2["position_coverage"] == v3["position_coverage"])
    return rows, {"compared": clean_compared, "identical": clean_identical,
                  "all_identical": clean_compared == clean_identical == 180}


def render(result: dict) -> str:
    """Indented JSON with one line per hunt row, so the evidence stays readable in a diff."""
    head = json.dumps({k: v for k, v in result.items() if k != "rows"}, indent=2)
    rows = ",\n".join("    " + json.dumps(row) for row in result["rows"])
    return f'{head[:-2]},\n  "rows": [\n{rows}\n  ]\n}}\n'


def head_commit(root: Path) -> str | None:
    done = subprocess.run(["git", "rev-parse", "HEAD"], cwd=root, capture_output=True, text=True)
    return done.stdout.strip() if done.returncode == 0 else None


def run(root: Path = ROOT) -> dict:
    protocol = load_protocol(root)
    maps = {"default": [default_layout()], "held_out": held_out_layouts(root)}
    if len(maps["held_out"]) != 8:
        raise ValueError("expected eight frozen held-out maps")
    references = committed_references(root)
    started = datetime.now(UTC)
    rows, clean_loop = run_hunts(protocol, maps)
    control = controls(rows, references)
    result = {
        "protocol": protocol["name"], "protocol_file": PROTOCOL_REL,
        "protocol_sha256": digest(root / PROTOCOL_REL),
        "protocol_written_at_utc": protocol["written_at_utc"],
        "code_commit": head_commit(root),
        "started_at_utc": started.isoformat(timespec="seconds"),
        "finished_at_utc": datetime.now(UTC).isoformat(timespec="seconds"),
        "hunts": len(rows),
        "acceptance": acceptance(rows, protocol, control, clean_loop),
        "controls": control, "clean_loop_v2_vs_v3": clean_loop,
        "summary": summarize(rows, protocol), "rows": rows,
    }
    saved_path = root / EVIDENCE_REL
    if saved_path.exists():
        saved = json.loads(saved_path.read_text())
        result["matches_committed_evidence"] = all(
            result[name] == saved[name] for name in ("rows", "summary", "acceptance", "controls"))
    output = root / OUTPUT_REL
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(render(result))
    return result


if __name__ == "__main__":
    outcome = run()
    print(json.dumps({key: outcome.get(key) for key in (
        "started_at_utc", "finished_at_utc", "hunts", "acceptance", "controls",
        "clean_loop_v2_vs_v3", "matches_committed_evidence")}, indent=2))
    print(json.dumps({"totals": outcome["summary"]["totals"],
                      "clean_v3_reached_goal": outcome["summary"]["clean_v3_reached_goal"],
                      "reward_fault_exposure_v3": outcome["summary"]["reward_fault_exposure_v3"],
                      "paired_differences": {name: len(item["differences"]) for name, item in
                                             outcome["summary"]["paired_v3_vs_v2"].items()}}, indent=2))
