"""Seeded train, validation and test GridQuest layouts, with leakage checks for each set.

Candidates come from the frozen ``playtest.levels`` generator and validator, which this module only
imports. A candidate is accepted in generation order, before any agent has played it, unless it is
invalid, or an exact or near duplicate of anything already accepted or of a reserved layout (the
default maps, the eight earlier bug-hunt maps, or the disclosed smoke maps).

Near-duplicate rule: two layouts of the same size whose labelled cells (wall, hazard, start, key,
coin, door, goal) differ in at most ``NEAR_DUPLICATE_CELLS`` cells, under any of the eight rotations
and reflections of the square. Defect-only fields are ignored, since clean play cannot see them.
"""
from __future__ import annotations

import hashlib
import json
import random

import numpy as np

from envs.gridquest import Layout, default_layout
from envs.gridquest_views import shortest_solution
from playtest.levels import _candidate, generate, validate

GENERATOR_VERSION = "gridquest-transfer-v1"
NEAR_DUPLICATE_CELLS = 4
SIZES = (6, 7, 8)
EARLIER_SEED, EARLIER_COUNT = 20261020, 8
LABELS = {"walls": 1, "hazards": 2, "start": 3, "key_at": 4, "coin_at": 5, "door_at": 6, "goal_at": 7}


def content(layout: Layout) -> dict:
    return {key: value for key, value in layout.to_json().items()
            if key not in ("wall_gap", "softlock_at")}


def content_hash(layout: Layout) -> str:
    """Hash what clean play depends on; maps differing only in defect cells hash the same."""
    return hashlib.sha256(json.dumps(content(layout), sort_keys=True,
                                     separators=(",", ":")).encode()).hexdigest()


def labelled_grid(layout: Layout) -> np.ndarray:
    grid = np.zeros((layout.size, layout.size), dtype=np.int8)
    for name, label in LABELS.items():
        cells = getattr(layout, name)
        for x, y in cells if isinstance(cells, frozenset) else (cells,):
            grid[y, x] = label
    return grid


def symmetries(grid: np.ndarray) -> np.ndarray:
    """The eight rotations and reflections of a square grid, stacked; index 0 is the grid itself."""
    turns = [np.rot90(grid, turn) for turn in range(4)]
    return np.stack(turns + [np.fliplr(turn) for turn in turns])


class LayoutRegistry:
    """Everything accepted so far, by size, for exact and near-duplicate rejection."""

    def __init__(self, threshold: int = NEAR_DUPLICATE_CELLS):
        self.threshold = threshold
        self.grids: dict[int, list[np.ndarray]] = {}
        self.hashes: set[str] = set()

    def distance(self, layout: Layout) -> int | None:
        """Fewest differing cells to any registered layout of its size, under symmetry."""
        known = self.grids.get(layout.size)
        if not known:
            return None
        stack = np.stack(known)[:, None]
        return int((stack != symmetries(labelled_grid(layout))[None]).sum(axis=(2, 3)).min())

    def conflict(self, layout: Layout) -> str | None:
        if content_hash(layout) in self.hashes:
            return "exact"
        distance = self.distance(layout)
        return "near" if distance is not None and distance <= self.threshold else None

    def add(self, layout: Layout) -> None:
        self.hashes.add(content_hash(layout))
        self.grids.setdefault(layout.size, []).append(labelled_grid(layout))


def _empty_rejections() -> dict[str, int]:
    return {"exact": 0, "near": 0, "invalid": 0, "unsolvable": 0}


def draw_set(rng: random.Random, count: int, registry: LayoutRegistry, rejected: dict) -> list[Layout]:
    """Accept ``count`` layouts in draw order, cycling through sizes; rejections are counted."""
    accepted: list[Layout] = []
    while len(accepted) < count:
        layout = _candidate(rng, SIZES[len(accepted) % len(SIZES)])
        if validate(layout):
            rejected["invalid"] += 1
        elif shortest_solution(layout) is None:
            rejected["unsolvable"] += 1
        else:
            conflict = registry.conflict(layout)
            if conflict:
                rejected[conflict] += 1
            else:
                registry.add(layout)
                accepted.append(layout)
    return accepted


def reserved_layouts(smoke_seed: int | None = None, smoke_count: int = 0) -> dict[str, list[Layout]]:
    """Layouts no train, validation or test set may match: already seen ones and smoke maps."""
    reserved = {"default": [default_layout(size) for size in SIZES],
                "earlier_eight": generate(EARLIER_SEED, EARLIER_COUNT)}
    if smoke_count:
        reserved["smoke"] = draw_set(random.Random(smoke_seed), smoke_count, LayoutRegistry(),
                                     _empty_rejections())
    return reserved


def build_sets(plan: dict) -> dict:
    """Generate every set from ``plan``: {sets: {name: {seed, count}}, smoke_seed, smoke_count}."""
    registry = LayoutRegistry(plan.get("near_duplicate_cells", NEAR_DUPLICATE_CELLS))
    reserved = reserved_layouts(plan.get("smoke_seed"), plan.get("smoke_count", 0))
    for layouts in reserved.values():
        for layout in layouts:
            registry.add(layout)
    sets, rejected = {}, {}
    for name, spec in plan["sets"].items():
        rejected[name] = _empty_rejections()
        sets[name] = draw_set(random.Random(spec["seed"]), spec["count"], registry, rejected[name])
    return {"sets": sets, "rejected": rejected, "reserved": reserved}


def set_hash(layouts: list[Layout]) -> str:
    return hashlib.sha256(json.dumps([content_hash(layout) for layout in layouts],
                                     separators=(",", ":")).encode()).hexdigest()


def _min_distance(first: list, second: list, same: bool) -> int | None:
    best = None
    for size in SIZES:
        left = [grid for grid, side in first if side == size]
        right = [grid[0] for grid, side in second if side == size]
        if not left or not right:
            continue
        right_stack = np.stack(right)
        for index, grid in enumerate(left):
            others = right_stack[index + 1:] if same else right_stack
            if not len(others):
                continue
            value = int((others[:, None] != grid[None]).sum(axis=(2, 3)).min())
            best = value if best is None else min(best, value)
    return best


def disjointness_report(sets: dict[str, list[Layout]], reserved: dict[str, list[Layout]],
                        threshold: int = NEAR_DUPLICATE_CELLS) -> dict:
    """Recheck, from the finished sets alone, that no two groups share a layout or a near copy."""
    groups = {**{f"reserved:{name}": layouts for name, layouts in reserved.items()}, **sets}
    grids = {name: [(symmetries(labelled_grid(layout)), layout.size) for layout in layouts]
             for name, layouts in groups.items()}
    report = {"threshold_cells": threshold, "pairs": {}, "within": {}}
    names = list(groups)
    for position, first in enumerate(names):
        hashes = {content_hash(layout) for layout in groups[first]}
        report["within"][first] = {"layouts": len(groups[first]), "distinct_hashes": len(hashes),
                                   "min_cell_distance": _min_distance(grids[first], grids[first], True)}
        for second in names[position + 1:]:
            if first.startswith("reserved:") and second.startswith("reserved:"):
                continue
            shared = hashes & {content_hash(layout) for layout in groups[second]}
            report["pairs"][f"{first}|{second}"] = {
                "shared_hashes": len(shared),
                "min_cell_distance": _min_distance(grids[first], grids[second], False)}
    pairs_clear = all(item["shared_hashes"] == 0 and (item["min_cell_distance"] is None
                      or item["min_cell_distance"] > threshold) for item in report["pairs"].values())
    within_clear = all(report["within"][name]["layouts"] == report["within"][name]["distinct_hashes"]
                       and (report["within"][name]["min_cell_distance"] is None
                            or report["within"][name]["min_cell_distance"] > threshold)
                       for name in sets)
    report["all_disjoint"] = pairs_clear and within_clear
    return report
