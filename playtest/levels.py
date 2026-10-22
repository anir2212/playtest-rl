"""Seeded generator and validator for held-out GridQuest maps.

Each generated map moves every object and both location-bound defects: the
off-map cell a ``wall_gap`` opens onto and the ``softlock`` trap. Validation
checks, by breadth-first search over (position, key) states, that the key, coin,
softlock trap and wall-gap edge are reachable without the key and that the goal
is reachable with it. The original map is excluded.
"""
from __future__ import annotations

import hashlib
import json
import random
from collections import deque

from envs.gridquest import DIRECTIONS, Layout, default_layout

GENERATOR_VERSION = "unseen-v1"


def reachable(layout: Layout, with_key: bool) -> set[tuple[int, int]]:
    """Cells reachable from the start; the door is passable only when ``with_key``."""
    seen = {layout.start}
    queue = deque([layout.start])
    while queue:
        x, y = queue.popleft()
        for dx, dy in DIRECTIONS:
            nxt = (x + dx, y + dy)
            if not (0 <= nxt[0] < layout.size and 0 <= nxt[1] < layout.size):
                continue
            if nxt in layout.walls or (nxt == layout.door_at and not with_key) or nxt in seen:
                continue
            seen.add(nxt)
            queue.append(nxt)
    return seen


def validate(layout: Layout) -> list[str]:
    """Return every rule the layout breaks (empty when it is playable)."""
    problems = []
    inside = [layout.start, layout.key_at, layout.coin_at, layout.door_at, layout.goal_at,
              layout.softlock_at, *layout.walls, *layout.hazards]
    if any(not (0 <= x < layout.size and 0 <= y < layout.size) for x, y in inside):
        problems.append("a cell lies outside the map")
    specials = [layout.start, layout.key_at, layout.coin_at, layout.door_at, layout.goal_at]
    if len(set(specials)) != len(specials):
        problems.append("start, key, coin, door and goal must differ")
    if set(specials + [layout.softlock_at]) & (layout.walls | layout.hazards):
        problems.append("a special cell is a wall or hazard")
    if layout.softlock_at in specials:
        problems.append("the softlock trap overlaps a special cell")
    if layout.door_at not in {(layout.goal_at[0] + dx, layout.goal_at[1] + dy)
                              for dx, dy in DIRECTIONS}:
        problems.append("the door must be next to the goal")
    no_key, with_key = reachable(layout, False), reachable(layout, True)
    for name in ("key_at", "coin_at", "softlock_at"):
        if getattr(layout, name) not in no_key:
            problems.append(f"{name} is unreachable without the key")
    if layout.goal_at not in with_key or layout.key_at not in no_key:
        problems.append("the goal is unreachable with the key")
    gx, gy = layout.wall_gap
    if 0 <= gx < layout.size and 0 <= gy < layout.size:
        problems.append("the wall gap must lead off the map")
    edge = [(gx + dx, gy + dy) for dx, dy in DIRECTIONS
            if 0 <= gx + dx < layout.size and 0 <= gy + dy < layout.size]
    if len(edge) != 1 or edge[0] not in no_key:
        problems.append("the wall gap must touch exactly one reachable edge cell")
    return problems


def _candidate(rng: random.Random, size: int) -> Layout:
    cells = [(x, y) for x in range(size) for y in range(size)]
    start = rng.choice(cells)
    goal = rng.choice([cell for cell in cells if cell != start])
    door = rng.choice([(goal[0] + dx, goal[1] + dy) for dx, dy in DIRECTIONS
                       if 0 <= goal[0] + dx < size and 0 <= goal[1] + dy < size
                       and (goal[0] + dx, goal[1] + dy) != start])
    free = [cell for cell in cells if cell not in (start, goal, door)]
    rng.shuffle(free)
    key, coin, softlock = free[0], free[1], free[2]
    walls = frozenset(free[3:3 + rng.randint(size // 2, size)])
    hazards = frozenset(free[3 + len(walls):3 + len(walls) + rng.randint(1, 2)])
    edge = [(x, y) for x, y in cells if x in (0, size - 1) or y in (0, size - 1)]
    ex, ey = rng.choice(edge)
    outward = [(ex + dx, ey + dy) for dx, dy in DIRECTIONS
               if not (0 <= ex + dx < size and 0 <= ey + dy < size)]
    return Layout(size=size, start=start, key_at=key, coin_at=coin, door_at=door, goal_at=goal,
                  walls=walls, hazards=hazards, wall_gap=rng.choice(outward), softlock_at=softlock)


def generate(seed: int, count: int, sizes: tuple[int, ...] = (6, 7, 8)) -> list[Layout]:
    """Draw candidates until ``count`` valid, distinct, non-original maps exist."""
    rng = random.Random(seed)
    originals = {default_layout(size) for size in sizes}
    levels: list[Layout] = []
    while len(levels) < count:
        layout = _candidate(rng, sizes[len(levels) % len(sizes)])
        if not validate(layout) and layout not in originals and layout not in levels:
            levels.append(layout)
    return levels


def manifest(seed: int, count: int) -> dict:
    levels = [layout.to_json() for layout in generate(seed, count)]
    body = {"generator": GENERATOR_VERSION, "seed": seed, "count": count, "levels": levels}
    body["sha256"] = hashlib.sha256(
        json.dumps(body, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
    return body


def render(layout: Layout) -> str:
    """ASCII map: S start, K key, $ coin, D door, G goal, # wall, ^ hazard, x trap, o gap edge."""
    gx, gy = layout.wall_gap
    edge = {(gx + dx, gy + dy) for dx, dy in DIRECTIONS}
    marks = {layout.start: "S", layout.key_at: "K", layout.coin_at: "$", layout.door_at: "D",
             layout.goal_at: "G", layout.softlock_at: "x"}
    return "\n".join("".join(marks.get((x, y)) or ("#" if (x, y) in layout.walls else
                                                    "^" if (x, y) in layout.hazards else
                                                    "o" if (x, y) in edge else ".")
                             for x in range(layout.size)) for y in range(layout.size))
