"""Predeclared bug-hunt evaluation on held-out GridQuest maps.

Two steps, run in order and never merged:

1. ``--write-protocol`` freezes the map manifest (seeded generator, SHA-256),
   policies, defects, seeds, step budget and metrics in
   ``evidence/unseen_levels_protocol.json``. It refuses to overwrite.
2. ``--run`` regenerates the maps, refuses to start if their hash differs from
   the protocol, then hunts every defect with every policy on every map and
   seed, plus a clean (defect-free) copy of every map to count false alarms.
   Results go to ``evidence/unseen_levels_results.json``.

The contract oracle (``playtest/oracle.py``) is built from each map's own
declared walls, door, goal and size and never reads ``info["bug_detected"]``.
Misses are right-censored at the step budget.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from datetime import UTC, datetime
from pathlib import Path
from statistics import mean, median

from envs.gridquest import Layout
from playtest.bug_hunt import BUGS, POLICIES, hunt
from playtest.levels import manifest

ROOT = Path(__file__).resolve().parents[1]
PROTOCOL_PATH = ROOT / "evidence" / "unseen_levels_protocol.json"
RESULTS_PATH = ROOT / "evidence" / "unseen_levels_results.json"
BASELINE_PATH = ROOT / "evidence" / "playtest_contract_oracle.json"
PROTOCOL = {
    "name": "gridquest-unseen-levels-v1",
    "level_seed": 20261020,
    "level_count": 8,
    "sizes": [6, 7, 8],
    "policies": list(POLICIES),
    "defects": list(BUGS),
    "hunt_seeds": [0, 1, 2, 3, 4],
    "max_steps": 1200,
    "primary_metrics": [
        "detections out of maps x seeds (40) per defect and policy, with a Wilson 95% interval",
        "mean and median discovery step among detections only; misses right-censored at 1200",
        ("false alarms on the clean copy of every map, per policy (any nonzero count is an "
         "oracle failure to report, not to tune away)"),
    ],
    "comparison": "Descriptive only: the original-map contract run (5 seeds, one map) in "
                  "evidence/playtest_contract_oracle.json. No significance test; the original "
                  "map was the development map for every policy.",
    "tuning": "None. Policies, oracle and budget are frozen as of this protocol.",
}


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def wilson(successes: int, n: int, z: float = 1.959964) -> list[float]:
    if n == 0:
        return [0.0, 0.0]
    p = successes / n
    centre = (p + z * z / (2 * n)) / (1 + z * z / n)
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / (1 + z * z / n)
    return [round(max(0.0, centre - half), 4), round(min(1.0, centre + half), 4)]


def write_protocol(path: Path = PROTOCOL_PATH) -> dict:
    if path.exists():
        raise FileExistsError(f"{path} exists; a predeclared protocol is never overwritten")
    levels = manifest(PROTOCOL["level_seed"], PROTOCOL["level_count"])
    body = {**PROTOCOL, "written_at_utc": datetime.now(UTC).isoformat(timespec="seconds"),
            "levels": levels, "baseline_sha256": sha256(BASELINE_PATH)}
    path.write_text(json.dumps(body, indent=2) + "\n", encoding="utf-8")
    return body


def summarize(rows: list[dict], seeds: int, maps: int) -> dict:
    n = seeds * maps
    cells = []
    for defect in BUGS:
        for policy in POLICIES:
            group = [row for row in rows if row["defect"] == defect and row["policy"] == policy]
            found = [row["time_to_discovery"] for row in group
                     if row["time_to_discovery"] is not None]
            kinds: dict[str, int] = {}
            for row in group:
                if row["detected"]:
                    kinds[row["detected"]] = kinds.get(row["detected"], 0) + 1
            cells.append({
                "defect": defect, "policy": policy, "detected": len(found), "n": n,
                "wilson_95": wilson(len(found), n),
                "mean_step_detected_only": round(mean(found), 1) if found else None,
                "median_step_detected_only": median(found) if found else None,
                "censored": n - len(found), "violation_kinds": kinds,
                "maps_with_any_detection": len({row["map"] for row in group
                                                if row["time_to_discovery"] is not None}),
            })
    clean = {policy: sum(1 for row in rows if row["defect"] is None and row["policy"] == policy
                         and row["detected"]) for policy in POLICIES}
    return {"cells": cells, "clean_false_alarms": clean, "clean_runs_per_policy": n}


def baseline_table() -> dict:
    data = json.loads(BASELINE_PATH.read_text(encoding="utf-8"))
    return {f"{row['bug']}|{row['policy']}": f"{row['discovered']}/{row['seeds']}"
            for row in data["summary"]}


def run(protocol_path: Path = PROTOCOL_PATH, out: Path = RESULTS_PATH) -> dict:
    protocol = json.loads(protocol_path.read_text(encoding="utf-8"))
    regenerated = manifest(protocol["level_seed"], protocol["level_count"])
    if regenerated["sha256"] != protocol["levels"]["sha256"]:
        raise RuntimeError("held-out maps differ from the predeclared protocol; refusing to run")
    if sha256(BASELINE_PATH) != protocol["baseline_sha256"]:
        raise RuntimeError("baseline evidence changed since the protocol was written")
    started = datetime.now(UTC)
    layouts = [Layout.from_json(level) for level in protocol["levels"]["levels"]]
    rows = []
    for index, layout in enumerate(layouts):
        for defect in [None, *protocol["defects"]]:
            for policy in protocol["policies"]:
                for seed in protocol["hunt_seeds"]:
                    result = hunt(defect, policy, seed, protocol["max_steps"], layout=layout)
                    rows.append({"map": index, "defect": defect, "policy": policy, "seed": seed,
                                 "time_to_discovery": result["time_to_discovery"],
                                 "detected": result["detected"],
                                 "position_coverage": round(result["position_coverage"], 4)})
    summary = summarize(rows, len(protocol["hunt_seeds"]), len(layouts))
    result = {
        "protocol": protocol["name"], "protocol_sha256": sha256(protocol_path),
        "protocol_written_at_utc": protocol["written_at_utc"],
        "started_at_utc": started.isoformat(timespec="seconds"),
        "finished_at_utc": datetime.now(UTC).isoformat(timespec="seconds"),
        "levels_sha256": regenerated["sha256"], "summary": summary,
        "original_map_contract_run": baseline_table(), "rows": rows,
    }
    out.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    action = parser.add_mutually_exclusive_group(required=True)
    action.add_argument("--write-protocol", action="store_true")
    action.add_argument("--run", action="store_true")
    args = parser.parse_args()
    if args.write_protocol:
        body = write_protocol()
        print(json.dumps({"written": str(PROTOCOL_PATH.relative_to(ROOT)),
                          "levels_sha256": body["levels"]["sha256"]}, indent=2))
        return
    result = run()
    table = {f"{cell['defect']}|{cell['policy']}": f"{cell['detected']}/{cell['n']}"
             for cell in result["summary"]["cells"]}
    print(json.dumps({"detections": table,
                      "clean_false_alarms": result["summary"]["clean_false_alarms"]}, indent=2))


if __name__ == "__main__":
    main()
