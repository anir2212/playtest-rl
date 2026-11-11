"""Run the separately committed gridquest-layout-transfer-v1 protocol: PPO on unseen GridQuest maps.

Refuses to start if the protocol file, a frozen input, a layout-set hash or the disjointness check
differs from what was committed. The two phases stay apart by construction: the search jobs only
ever receive training and validation layouts, and the test layouts are first handed to a job after
the validation selection is fixed. Writes only ``runs/gridquest_transfer.json``; the committed copy
of the first scored run is ``evidence/gridquest_transfer_results.json``, and later runs report
whether they match it.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import multiprocessing
import subprocess
import time
from concurrent.futures import ProcessPoolExecutor
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import torch

from agents import layout_ppo
from envs.gridquest import GridQuest, Layout
from envs.gridquest_views import path_score, shortest_solution
from playtest.bug_hunt import _scripted_action
from playtest.transfer_layouts import (build_sets, disjointness_report, reserved_layouts, set_hash)

ROOT = Path(__file__).resolve().parents[1]
PROTOCOL_REL = "evidence/gridquest_transfer_protocol.json"
EVIDENCE_REL = "evidence/gridquest_transfer_results.json"
OUTPUT_REL = "runs/gridquest_transfer.json"
PROTOCOL_SHA256 = "4f835e902ab6d807220c5a3278ded69d98270a8d57df65c5fa819cd468879a60"
COMPARED = ("selection", "rows", "references", "summary", "acceptance", "disjointness")


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_protocol(root: Path = ROOT) -> dict:
    path = root / PROTOCOL_REL
    if digest(path) != PROTOCOL_SHA256:
        raise RuntimeError("layout transfer protocol differs from its committed hash")
    protocol = json.loads(path.read_text())
    if (protocol["name"] != "gridquest-layout-transfer-v1"
            or protocol["observation"]["primary"] != "egocentric"
            or set(protocol["layout_plan"]["sets"]) != {"train", "validation", "test"}
            or len(protocol["final_seeds"]) < 5
            or set(protocol["final_seeds"]) & set(protocol["search_seeds"])):
        raise ValueError("layout transfer protocol structure changed")
    for rel, frozen_hash in protocol["frozen_inputs_sha256"].items():
        if digest(root / rel) != frozen_hash:
            raise RuntimeError(f"frozen input changed: {rel}")
    return protocol


def load_sets(protocol: dict) -> tuple[dict[str, list[Layout]], dict]:
    """Regenerate every layout set; refuse on a changed hash or any overlap."""
    plan = protocol["layout_plan"]
    built = build_sets(plan)
    for name, expected in protocol["layout_set_sha256"].items():
        if set_hash(built["sets"][name]) != expected:
            raise RuntimeError(f"layout set differs from the protocol: {name}")
    if built["rejected"] != protocol["generation_rejections"]:
        raise RuntimeError("layout generation rejections differ from the protocol")
    report = disjointness_report(built["sets"], built["reserved"], plan["near_duplicate_cells"])
    if not report["all_disjoint"] or report != protocol["disjointness"]:
        raise RuntimeError("layout sets are not disjoint as the protocol recorded")
    return built["sets"], report


def training_block(protocol: dict, sets: dict, slot: int, count: int) -> list[Layout]:
    """Each seed slot owns a separate block of the training set; smaller counts are its prefix."""
    width = max(protocol["training_layout_counts"])
    block = sets["train"][slot * width:(slot + 1) * width]
    if len(block) != width or count > width:
        raise ValueError("training set too small for this seed slot")
    return block[:count]


def train_job(job: dict) -> dict:
    torch.set_num_threads(1)
    started = time.monotonic()
    settings = job["settings"]
    model, info = layout_ppo.train(job["layouts"], job["train_seed"], job["steps"], view=job["view"],
                                   reward_mode=job["reward_mode"], **settings)
    evaluations = {}
    for name, layouts in job["eval_sets"].items():
        evaluations[name] = {}
        for mode, episodes in job["eval_modes"].items():
            outcome = layout_ppo.evaluate(model, layouts, job["eval_seed"], episodes, view=job["view"],
                                          mode=mode, reward_mode=job["reward_mode"])
            evaluations[name][mode] = {"success": [round(v, 4) for v in outcome["success"]],
                                       "spl": [round(v, 4) for v in outcome["spl"]]}
    history = info["history"]
    tail = history[-max(1, len(history) // 10):]
    row = {key: job[key] for key in ("phase", "condition", "view", "count", "slot", "train_seed",
                                     "config", "steps", "eval_seed")}
    row.update(param_sha256_16=layout_ppo.fingerprint(model), consumed_steps=info["steps"],
               train_tail_success=round(sum(item["successes"] for item in tail)
                                        / max(1, sum(item["episodes"] for item in tail)), 4),
               evaluations=evaluations)
    return {"row": row, "seconds": round(time.monotonic() - started, 1)}


def _job(protocol: dict, phase: str, condition: str, view: str, count: int, slot: int, train_seed: int,
         config: str, sets: dict, eval_sets: dict) -> dict:
    return {"phase": phase, "condition": condition, "view": view, "count": count, "slot": slot,
            "train_seed": train_seed, "config": config, "steps": protocol["steps"],
            "reward_mode": protocol["reward_mode"], "settings": {**protocol["fixed_ppo"],
                                                                 **protocol["search_space"][config]},
            "layouts": training_block(protocol, sets, slot, count), "eval_sets": eval_sets,
            "eval_modes": protocol["evaluation"]["episodes_per_layout"],
            "eval_seed": protocol["evaluation"]["seed_base"] + train_seed}


def search_jobs(protocol: dict, sets: dict) -> list[dict]:
    """Every candidate setting on the largest training count; no test layout is passed."""
    count, jobs = max(protocol["training_layout_counts"]), []
    for config in protocol["search_space"]:
        for slot, seed in enumerate(protocol["search_seeds"]):
            jobs.append(_job(protocol, "search", f"egocentric_{count}", "egocentric", count, slot, seed,
                             config, sets, {"validation": sets["validation"]}))
    return jobs


def select(rows: list[dict], protocol: dict) -> dict:
    """Highest mean validation score across search seeds; ties go to the earlier listed setting."""
    metric, mode = protocol["selection"]["metric"], protocol["selection"]["mode"]
    table = {}
    for config in protocol["search_space"]:
        group = sorted((row for row in rows if row["config"] == config), key=lambda row: row["train_seed"])
        if [row["train_seed"] for row in group] != protocol["search_seeds"]:
            raise ValueError(f"search rows incomplete: {config}")
        scores = [float(np.mean(row["evaluations"]["validation"][mode][metric])) for row in group]
        table[config] = {"seed_scores": [round(value, 4) for value in scores],
                         "mean": round(float(np.mean(scores)), 4)}
    chosen = max(table, key=lambda name: (table[name]["mean"],
                                          -list(protocol["search_space"]).index(name)))
    return {"metric": metric, "mode": mode, "table": table, "chosen": chosen}


def final_jobs(protocol: dict, sets: dict, chosen: str, secondary: list[Layout]) -> list[dict]:
    """Chosen setting only: each count and control on each final seed, evaluated on every set."""
    offset, jobs = len(protocol["search_seeds"]), []
    for slot, seed in enumerate(protocol["final_seeds"], offset):
        for entry in protocol["conditions"]:
            layouts = training_block(protocol, sets, slot, entry["count"])
            jobs.append(_job(protocol, "final", entry["name"], entry["view"], entry["count"], slot, seed,
                             chosen, sets, {"own_training": layouts, "validation": sets["validation"],
                                            "test": sets["test"], "earlier_eight": secondary}))
    return jobs


def scripted_reference(layouts: list[Layout]) -> dict:
    """Upper reference: the existing key-then-goal search played in the real game on each map.

    Scored with the protocol's own SPL, ``shortest / max(taken, shortest)`` on success, so a
    search that solves a map the long way round is not credited as optimal. The count of maps
    finished in exactly the shortest number of moves is kept beside it as a description.
    """
    solved, on_shortest, scores = 0, 0, []
    for layout in layouts:
        env = GridQuest(layout=layout)
        env.reset()
        info = {"success": False}
        while True:
            *_, terminated, truncated, info = env.step(_scripted_action(env))
            if terminated or truncated:
                break
        won = bool(info["success"])
        shortest = shortest_solution(layout)
        solved += won
        on_shortest += won and env.steps == shortest
        scores.append(path_score(won, shortest, env.steps))
    return {"layouts": len(layouts), "solved": solved, "on_shortest_path": on_shortest,
            "spl": round(float(np.mean(scores)), 4)}


def random_reference(layouts: list[Layout], seed: int, episodes: int) -> dict:
    outcome = layout_ppo.evaluate(None, layouts, seed, episodes)
    return {"success": [round(v, 4) for v in outcome["success"]],
            "spl": [round(v, 4) for v in outcome["spl"]]}


def crossed_interval(matrix: np.ndarray, rng: np.random.Generator, resamples: int) -> list[float]:
    """Bootstrap seeds and layouts together, so neither is treated as fixed."""
    seeds, layouts = matrix.shape
    draws = np.empty(resamples)
    for index in range(resamples):
        draws[index] = matrix[rng.integers(0, seeds, seeds)][:, rng.integers(0, layouts, layouts)].mean()
    return [round(float(value), 4) for value in np.quantile(draws, [0.025, 0.975])]


def contrast_interval(first: np.ndarray, second: np.ndarray, rng: np.random.Generator,
                      resamples: int) -> list[float]:
    """Mean of first minus second: layouts paired across arms, seeds drawn apart for each arm."""
    layouts = first.shape[1]
    if second.shape[1] != layouts:
        raise ValueError("arms must share layouts")
    draws = np.empty(resamples)
    for index in range(resamples):
        chosen = rng.integers(0, layouts, layouts)
        one = first[rng.integers(0, len(first), len(first))][:, chosen].mean()
        two = second[rng.integers(0, len(second), len(second))][:, chosen].mean()
        draws[index] = one - two
    return [round(float(value), 4) for value in np.quantile(draws, [0.025, 0.975])]


def gap_interval(train: np.ndarray, test: np.ndarray, rng: np.random.Generator,
                 resamples: int) -> list[float]:
    """Train minus test for the same agents: seeds shared, each layout set drawn on its own."""
    if len(train) != len(test):
        raise ValueError("gap needs the same seeds on both sets")
    seeds, train_layouts, test_layouts = len(train), train.shape[1], test.shape[1]
    draws = np.empty(resamples)
    for index in range(resamples):
        chosen = rng.integers(0, seeds, seeds)
        draws[index] = (train[chosen][:, rng.integers(0, train_layouts, train_layouts)].mean()
                        - test[chosen][:, rng.integers(0, test_layouts, test_layouts)].mean())
    return [round(float(value), 4) for value in np.quantile(draws, [0.025, 0.975])]


def matrix(rows: list[dict], condition: str, set_name: str, mode: str, metric: str) -> np.ndarray:
    group = sorted((row for row in rows if row["condition"] == condition),
                   key=lambda row: row["train_seed"])
    return np.array([row["evaluations"][set_name][mode][metric] for row in group], dtype=float)


def summarize(rows: list[dict], references: dict, protocol: dict, sets: dict) -> dict:
    boot = protocol["bootstrap"]
    rng = np.random.default_rng(boot["seed"])
    resamples = boot["resamples"]
    conditions = [entry["name"] for entry in protocol["conditions"]]
    random_rows = [{"condition": "random", "train_seed": seed, "evaluations": {
        name: {"sampled": references["random"][name][str(seed)]} for name in ("validation", "test", "earlier_eight")}}
        for seed in protocol["final_seeds"]]
    everyone = rows + random_rows
    cells = {}
    for condition in [*conditions, "random"]:
        cells[condition] = {}
        for set_name in ("own_training", "validation", "test", "earlier_eight"):
            if condition == "random" and set_name == "own_training":
                continue
            cells[condition][set_name] = {}
            for mode in protocol["evaluation"]["episodes_per_layout"]:
                if condition == "random" and mode != "sampled":
                    continue
                for metric in ("spl", "success"):
                    values = matrix(everyone, condition, set_name, mode, metric)
                    cells[condition][set_name][f"{mode}_{metric}"] = {
                        "mean": round(float(values.mean()), 4),
                        "ci95": crossed_interval(values, rng, resamples),
                        "seeds": values.shape[0], "layouts": values.shape[1],
                        "seed_means": [round(float(v), 4) for v in values.mean(axis=1)]}
    mode, metric = protocol["primary_metric"]["mode"], protocol["primary_metric"]["metric"]
    key = protocol["primary_contrast"]
    test = {name: matrix(everyone, name, "test", mode, metric) for name in [*conditions, "random"]}
    contrasts = {}
    for label, (first, second) in key["contrasts"].items():
        contrasts[label] = {"first": first, "second": second,
                            "difference": round(float(test[first].mean() - test[second].mean()), 4),
                            "ci95": contrast_interval(test[first], test[second], rng, resamples)}
    gaps = {}
    for name in conditions:
        train = matrix(rows, name, "own_training", mode, metric)
        gaps[name] = {"own_training": round(float(train.mean()), 4), "test": round(float(test[name].mean()), 4),
                      "gap": round(float(train.mean() - test[name].mean()), 4),
                      "ci95": gap_interval(train, test[name], rng, resamples)}
    sizes = np.array([layout.size for layout in sets["test"]])
    by_size = {name: {str(size): round(float(test[name][:, sizes == size].mean()), 4)
                      for size in sorted(set(sizes.tolist()))} for name in [*conditions, "random"]}
    lead = key["agent"]
    per_layout = test[lead].mean(axis=0)
    base = test["random"].mean(axis=0)
    return {"primary_metric": protocol["primary_metric"], "cells": cells, "contrasts": contrasts,
            "generalisation_gap": gaps, "test_by_size": by_size,
            "lead_agent_test_layouts": {
                "layouts": len(per_layout), "above_random": int((per_layout > base).sum()),
                "at_or_below_random": int((per_layout <= base).sum()),
                "spl_at_least_half": int((per_layout >= 0.5).sum()),
                "lowest_layout_spl": round(float(per_layout.min()), 4),
                "highest_layout_spl": round(float(per_layout.max()), 4)}}


def acceptance(summary: dict, protocol: dict) -> dict:
    rules = protocol["acceptance_rule"]
    contrasts = summary["contrasts"]
    margin = contrasts["agent_vs_random"]["ci95"][0] >= rules["agent_vs_random_lower_bound_at_least"]
    single = contrasts["agent_vs_single_layout"]["ci95"][0] > 0
    control = contrasts["agent_vs_absolute_control"]["ci95"][0] > 0
    return {"A_beats_random_by_margin": margin, "B_beats_single_layout_reference": single,
            "C_beats_absolute_control": control,
            "overall": "PASS" if margin and single and control else "FAIL"}


def render(result: dict) -> str:
    """Indented JSON with one line per training run, so the evidence stays readable in a diff."""
    head = json.dumps({k: v for k, v in result.items() if k != "rows"}, indent=2)
    rows = ",\n".join("    " + json.dumps(row) for row in result["rows"])
    return f'{head[:-2]},\n  "rows": [\n{rows}\n  ]\n}}\n'


def head_commit(root: Path) -> str | None:
    done = subprocess.run(["git", "rev-parse", "HEAD"], cwd=root, capture_output=True, text=True)
    return done.stdout.strip() if done.returncode == 0 else None


def _pool_run(jobs: list[dict], workers: int, label: str) -> tuple[list[dict], list[float]]:
    context = multiprocessing.get_context("spawn")
    rows, seconds = [], []
    with ProcessPoolExecutor(max_workers=workers, mp_context=context) as pool:
        for done, outcome in enumerate(pool.map(train_job, jobs), 1):
            rows.append(outcome["row"])
            seconds.append(outcome["seconds"])
            print(f"{label} {done}/{len(jobs)} {outcome['row']['condition']} "
                  f"seed={outcome['row']['train_seed']} {outcome['seconds']}s", flush=True)
    return rows, seconds


def run(root: Path = ROOT, workers: int = 8) -> dict:
    protocol = load_protocol(root)
    sets, disjointness = load_sets(protocol)
    secondary = reserved_layouts()["earlier_eight"]
    started = datetime.now(UTC)
    search_rows, search_seconds = _pool_run(search_jobs(protocol, sets), workers, "search")
    selection = select(search_rows, protocol)
    print("selected", json.dumps(selection), flush=True)
    final_rows, final_seconds = _pool_run(final_jobs(protocol, sets, selection["chosen"], secondary),
                                          workers, "final")
    episodes = protocol["evaluation"]["episodes_per_layout"]["sampled"]
    references = {
        "scripted": {name: scripted_reference(sets[name]) for name in ("validation", "test")}
        | {"earlier_eight": scripted_reference(secondary)},
        "random": {name: {str(seed): random_reference(layouts, protocol["evaluation"]["seed_base"] + seed,
                                                      episodes)
                          for seed in protocol["final_seeds"]}
                   for name, layouts in (("validation", sets["validation"]), ("test", sets["test"]),
                                         ("earlier_eight", secondary))}}
    summary = summarize(final_rows, references, protocol, sets)
    result = {
        "protocol": protocol["name"], "protocol_file": PROTOCOL_REL,
        "protocol_sha256": digest(root / PROTOCOL_REL),
        "protocol_written_at_utc": protocol["written_at_utc"], "code_commit": head_commit(root),
        "started_at_utc": started.isoformat(timespec="seconds"),
        "finished_at_utc": datetime.now(UTC).isoformat(timespec="seconds"),
        "workers": workers, "job_seconds": {"search": search_seconds, "final": final_seconds},
        "disjointness": disjointness, "selection": selection, "acceptance": acceptance(summary, protocol),
        "summary": summary, "references": references, "rows": search_rows + final_rows,
    }
    saved_path = root / EVIDENCE_REL
    if saved_path.exists():
        saved = json.loads(saved_path.read_text())
        result["matches_committed_evidence"] = all(result[name] == saved[name] for name in COMPARED)
    output = root / OUTPUT_REL
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(render(result))
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--workers", type=int, default=8)
    args = parser.parse_args()
    outcome = run(workers=args.workers)
    print(json.dumps({key: outcome.get(key) for key in (
        "started_at_utc", "finished_at_utc", "selection", "acceptance", "matches_committed_evidence")},
        indent=2))
    print(json.dumps({"contrasts": outcome["summary"]["contrasts"],
                      "generalisation_gap": outcome["summary"]["generalisation_gap"],
                      "scripted": outcome["references"]["scripted"]}, indent=2))


if __name__ == "__main__":
    main()
