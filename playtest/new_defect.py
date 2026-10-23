"""Predeclared bug hunt for a defect type the frozen contract was not written around.

Two steps, run in order and never merged:

1. ``--write-protocol`` copies ``PROTOCOL`` to ``evidence/new_defect_protocol.json`` with
   the SHA-256 of the frozen held-out map protocol and the evidence it compares against.
   It refuses to overwrite.
2. ``--run`` refuses to start if that file differs from ``PROTOCOL``, a frozen input
   changed, or the regenerated held-out maps differ. It then hunts the new defect, and a
   clean copy of every map, with every policy, seed and oracle, and writes
   ``runs/new_defect.json``. The committed copy of the one predeclared run is
   ``evidence/new_defect_results.json``.

The held-out maps are read from ``evidence/unseen_levels_protocol.json``; that protocol,
its results and the original-map evidence are not modified. Neither oracle reads
``info["bug_detected"]``. Misses are right-censored at the step budget.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from datetime import UTC, datetime
from pathlib import Path
from statistics import mean, median

from envs.gridquest import Layout, default_layout
from playtest.bug_hunt import NEW_DEFECTS, ORACLES, POLICIES, hunt
from playtest.levels import manifest
from playtest.unseen import wilson

ROOT = Path(__file__).resolve().parents[1]
PROTOCOL_PATH = ROOT / "evidence" / "new_defect_protocol.json"
EVIDENCE_PATH = ROOT / "evidence" / "new_defect_results.json"
RUN_PATH = ROOT / "runs" / "new_defect.json"
FROZEN_INPUTS = {
    "unseen_protocol": "evidence/unseen_levels_protocol.json",
    "unseen_results": "evidence/unseen_levels_results.json",
    "original_map_contract_run": "evidence/playtest_contract_oracle.json",
}
REQUIRED_SEEDS = 5

PROTOCOL = {
    "name": "gridquest-new-defect-v1",
    "declared": "2026-10-23, before any hunt on the new defect",
    "defect": "coin_not_scored",
    "defect_description": "Entering the coin tile for the first time in an episode sets the "
                          "coin flag but does not add the declared 1 point. Every later score "
                          "stays inside the declared 0-11 range, so it is a plausible but wrong "
                          "value rather than an out-of-range one.",
    "why_this_defect": "Each earlier defect moves the player illegally, blocks a legal move "
                       "or pushes the score out of range. This one does none of those. The "
                       "frozen v1 contract checks bounds, the score range and movement "
                       "legality, so it was not written around this defect.",
    "maps": {
        "default": "the original GridQuest map (default_layout, size 6), the development map "
                   "for every policy",
        "held_out": "the eight frozen gridquest-unseen-v1 maps, read from "
                    "evidence/unseen_levels_protocol.json and regenerated; the run refuses if "
                    "their SHA-256 differs. The coin is already placed differently on each map.",
    },
    "policies": list(POLICIES),
    "hunt_seeds": [0, 1, 2, 3, 4],
    "max_steps": 1200,
    "oracles": {
        "v1": "playtest/oracle.observe_transition, unchanged since 18 October: out of bounds, "
              "score outside 0-11, blocked legal move or goal entry, illegal move",
        "v2": "playtest/oracle.check_transition, new: predicts position, key, coin and score "
              "after each move from the map's declared size, walls, door, key, coin and goal "
              "and the declared scoring (coin 1 point, goal 10 points), then compares them with "
              "the reported state. It was written in the same session as the defect, so it is "
              "not a blind test; v1 is the blind test.",
    },
    "prediction": "From reading the code before any hunt: v1 detects 0 of 45 hunts per policy, "
                  "because the coin point is the only change and the score stays in range. "
                  "v2 flags score_mismatch at the first coin entry. Reported whatever happens.",
    "detection_rule": "A hunt detects the defect at the first step where the oracle reports any "
                      "violation on the defective map. Misses are right-censored at 1,200 steps "
                      "and never given a discovery time. The violation kind is reported; "
                      "a kind other than score_mismatch under v2 still counts as a detection "
                      "and is listed.",
    "metrics": [
        "detections out of 5 (default map) and out of 40 (8 held-out maps x 5 seeds) per "
        "policy and oracle, with a Wilson 95% interval",
        "mean and median discovery step among detections only; number censored",
        "false alarms on a clean copy of every map, per policy and oracle, out of 5 and 40",
    ],
    "reading_rules": [
        "Any v2 false alarm on a clean map is an oracle failure: it is reported, and v2's "
        "detections on that map set are not presented as clean evidence.",
        "A nonzero v1 detection count means the prediction was wrong and is reported as such.",
        "Policy differences are descriptive; no significance test is made.",
    ],
    "consistency_check": "The v1 clean-copy rows on the held-out maps must equal the clean rows "
                         "in evidence/unseen_levels_results.json (discovery step, violation and "
                         "coverage), which shows the harness change left v1 hunts unchanged.",
    "tuning": "None. Defect, maps, policies, oracles, seeds and budget are frozen by this "
              "protocol; misses are reported, not re-run.",
}


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def check_protocol(protocol: dict) -> None:
    seeds = protocol["hunt_seeds"]
    if len(seeds) != REQUIRED_SEEDS or len(set(seeds)) != REQUIRED_SEEDS:
        raise ValueError(f"need {REQUIRED_SEEDS} distinct hunt seeds, got {seeds}")
    if protocol["defect"] not in NEW_DEFECTS:
        raise ValueError("the defect must be one the frozen protocols do not include")
    if set(protocol["oracles"]) != set(ORACLES) or set(protocol["policies"]) != set(POLICIES):
        raise ValueError("every oracle and policy must be declared")


def write_protocol(path: Path = PROTOCOL_PATH, protocol: dict = PROTOCOL, root: Path = ROOT) -> dict:
    if path.exists():
        raise FileExistsError(f"{path} exists; a predeclared protocol is never overwritten")
    check_protocol(protocol)
    body = {**protocol, "written_at_utc": datetime.now(UTC).isoformat(timespec="seconds"),
            "frozen_inputs_sha256": {name: sha256(root / rel) for name, rel in FROZEN_INPUTS.items()}}
    path.write_text(json.dumps(body, indent=2) + "\n", encoding="utf-8")
    return body


def load_protocol(path: Path = PROTOCOL_PATH, protocol: dict = PROTOCOL, root: Path = ROOT) -> dict:
    saved = json.loads(path.read_text(encoding="utf-8"))
    declared = {k: v for k, v in saved.items() if k not in ("written_at_utc", "frozen_inputs_sha256")}
    if declared != json.loads(json.dumps(protocol)):
        raise RuntimeError(f"{path} differs from the PROTOCOL constant; refusing to run")
    for name, rel in FROZEN_INPUTS.items():
        if sha256(root / rel) != saved["frozen_inputs_sha256"][name]:
            raise RuntimeError(f"{rel} changed since the protocol was written")
    check_protocol(saved)
    return saved


def held_out_layouts(root: Path = ROOT) -> list[Layout]:
    """The frozen held-out maps, refusing if the generator no longer reproduces them."""
    frozen = json.loads((root / FROZEN_INPUTS["unseen_protocol"]).read_text(encoding="utf-8"))
    regenerated = manifest(frozen["level_seed"], frozen["level_count"])
    if regenerated["sha256"] != frozen["levels"]["sha256"]:
        raise RuntimeError("held-out maps differ from the frozen protocol; refusing to run")
    return [Layout.from_json(level) for level in frozen["levels"]["levels"]]


def summarize(rows: list[dict], protocol: dict) -> dict:
    out = {}
    seeds = len(protocol["hunt_seeds"])
    for map_set in ("default", "held_out"):
        n_maps = len({r["map"] for r in rows if r["map_set"] == map_set})
        n = seeds * n_maps
        out[map_set] = {"n": n, "maps": n_maps, "cells": [], "clean_false_alarms": {}}
        for oracle in protocol["oracles"]:
            for policy in protocol["policies"]:
                group = [r for r in rows if r["map_set"] == map_set and r["oracle"] == oracle
                         and r["policy"] == policy]
                defect = [r for r in group if r["defect"] == protocol["defect"]]
                clean = [r for r in group if r["defect"] is None]
                if len(defect) != n or len(clean) != n:
                    raise ValueError(f"{map_set}/{oracle}/{policy}: expected {n} hunts each")
                found = [r["time_to_discovery"] for r in defect if r["time_to_discovery"] is not None]
                kinds: dict[str, int] = {}
                for r in defect:
                    if r["detected"]:
                        kinds[r["detected"]] = kinds.get(r["detected"], 0) + 1
                out[map_set]["cells"].append({
                    "oracle": oracle, "policy": policy, "detected": len(found), "n": n,
                    "wilson_95": wilson(len(found), n),
                    "mean_step_detected_only": round(mean(found), 1) if found else None,
                    "median_step_detected_only": median(found) if found else None,
                    "censored": n - len(found), "violation_kinds": kinds,
                    "maps_with_any_detection": len({r["map"] for r in defect
                                                    if r["time_to_discovery"] is not None})})
                out[map_set]["clean_false_alarms"][f"{oracle}|{policy}"] = sum(
                    1 for r in clean if r["detected"])
    return out


def unseen_consistency(rows: list[dict], root: Path = ROOT) -> dict:
    saved = json.loads((root / FROZEN_INPUTS["unseen_results"]).read_text(encoding="utf-8"))
    key = lambda r: (r["map"], r["policy"], r["seed"])  # noqa: E731
    reference = {key(r): r for r in saved["rows"] if r["defect"] is None}
    mine = [r for r in rows if r["map_set"] == "held_out" and r["oracle"] == "v1" and r["defect"] is None]
    same = sum(1 for r in mine if all(r[f] == reference[key(r)][f]
                                      for f in ("time_to_discovery", "detected", "position_coverage")))
    return {"compared": len(mine), "identical": same,
            "all_identical": len(mine) == len(reference) == same}


def run(protocol_path: Path = PROTOCOL_PATH, out: Path = RUN_PATH, protocol_constant: dict = PROTOCOL,
        root: Path = ROOT, compare_with: Path | None = EVIDENCE_PATH) -> dict:
    protocol = load_protocol(protocol_path, protocol_constant, root)
    maps = {"default": [default_layout()], "held_out": held_out_layouts(root)}
    started = datetime.now(UTC)
    rows = []
    for map_set, layouts in maps.items():
        for index, layout in enumerate(layouts):
            for defect in (None, protocol["defect"]):
                for oracle in protocol["oracles"]:
                    for policy in protocol["policies"]:
                        for seed in protocol["hunt_seeds"]:
                            result = hunt(defect, policy, seed, protocol["max_steps"],
                                          layout=layout, oracle=oracle)
                            rows.append({"map_set": map_set, "map": index, "defect": defect,
                                         "oracle": oracle, "policy": policy, "seed": seed,
                                         "time_to_discovery": result["time_to_discovery"],
                                         "detected": result["detected"],
                                         "position_coverage": round(result["position_coverage"], 4)})
    result = {
        "protocol": protocol["name"], "protocol_file": protocol_path.name,
        "protocol_sha256": sha256(protocol_path),
        "protocol_written_at_utc": protocol["written_at_utc"],
        "started_at_utc": started.isoformat(timespec="seconds"),
        "finished_at_utc": datetime.now(UTC).isoformat(timespec="seconds"),
        "summary": summarize(rows, protocol),
        "v1_clean_matches_unseen_evidence": unseen_consistency(rows, root),
        "rows": rows,
    }
    if compare_with is not None and compare_with.exists():
        saved = json.loads(compare_with.read_text(encoding="utf-8"))
        result["matches_committed_evidence"] = saved["rows"] == json.loads(json.dumps(rows))
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    return result


def headline(result: dict) -> dict:
    table = {}
    for map_set, data in result["summary"].items():
        for cell in data["cells"]:
            table[f"{map_set}|{cell['oracle']}|{cell['policy']}"] = (
                f"{cell['detected']}/{cell['n']}, mean step {cell['mean_step_detected_only']}")
    return {"detections": table,
            "clean_false_alarms": {m: d["clean_false_alarms"] for m, d in result["summary"].items()},
            "v1_clean_matches_unseen_evidence": result["v1_clean_matches_unseen_evidence"],
            "matches_committed_evidence": result.get("matches_committed_evidence")}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    action = parser.add_mutually_exclusive_group(required=True)
    action.add_argument("--write-protocol", action="store_true")
    action.add_argument("--run", action="store_true")
    args = parser.parse_args()
    if args.write_protocol:
        body = write_protocol()
        print(json.dumps({"written": str(PROTOCOL_PATH.relative_to(ROOT)),
                          "frozen_inputs_sha256": body["frozen_inputs_sha256"]}, indent=2))
        return
    result = run()
    print(f"{RUN_PATH.relative_to(ROOT)} written")
    print(json.dumps(headline(result), indent=2))


if __name__ == "__main__":
    main()
