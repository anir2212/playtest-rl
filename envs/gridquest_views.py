"""Observation views and a multi-layout wrapper for GridQuest; the base game is not changed.

``GridQuest`` itself reports one integer built from the absolute position and two flags. It never
says where the walls, key, door or goal are, so a policy reading it can only memorise one map.
This module adds, beside it, two array views and an environment that draws a layout per episode:

* ``egocentric``: the whole map as planes centred on the player, so the same local arrangement
  looks the same wherever it occurs and on whichever map. It shows walls and the off-map area as
  blocked cells, and the door, untaken key, goal, hazards and untaken coin. Defect-only cells
  (``softlock_at``, ``wall_gap``) are not shown.
* ``absolute``: the information the integer observation holds (position and two flags) as one-hot
  columns. It holds nothing about the layout and is kept as a negative control.
"""
from __future__ import annotations

import random
from collections import deque
from collections.abc import Sequence
from functools import lru_cache

import gymnasium as gym
import numpy as np
from gymnasium import spaces

from envs.gridquest import DIRECTIONS, GridQuest, Layout

MAX_SIZE = 8
RADIUS = MAX_SIZE - 1
WINDOW = 2 * RADIUS + 1
PLANES = ("blocked", "door", "key", "goal", "hazard", "coin", "has_key")
BLOCKED, DOOR, KEY, GOAL, HAZARD, COIN, HAS_KEY = range(len(PLANES))
EGOCENTRIC_SHAPE = (len(PLANES), WINDOW, WINDOW)
ABSOLUTE_SIZE = 2 * MAX_SIZE + 2
VIEWS = ("egocentric", "absolute")


@lru_cache(maxsize=None)
def _static_planes(layout: Layout) -> np.ndarray:
    """Read-only layout planes padded by the view radius, indexed [plane, y + RADIUS, x + RADIUS]."""
    side = layout.size + 2 * RADIUS
    planes = np.zeros((len(PLANES), side, side), dtype=np.float32)
    planes[BLOCKED] = 1.0
    planes[BLOCKED, RADIUS:RADIUS + layout.size, RADIUS:RADIUS + layout.size] = 0.0
    for plane, cells in ((BLOCKED, layout.walls), (DOOR, (layout.door_at,)),
                         (KEY, (layout.key_at,)), (GOAL, (layout.goal_at,)),
                         (HAZARD, layout.hazards), (COIN, (layout.coin_at,))):
        for x, y in cells:
            planes[plane, y + RADIUS, x + RADIUS] = 1.0
    planes.flags.writeable = False
    return planes


class ViewedGridQuest:
    """One GridQuest layout plus cached planes; returns array observations."""

    def __init__(self, layout: Layout, view: str = "egocentric", reward_mode: str = "sparse"):
        if view not in VIEWS:
            raise ValueError("unknown view")
        if layout.size > MAX_SIZE:
            raise ValueError(f"layout size above the view limit of {MAX_SIZE}")
        self.layout, self.view = layout, view
        self.env = GridQuest(layout=layout, reward_mode=reward_mode)
        self.planes = _static_planes(layout)

    def observe(self) -> np.ndarray:
        x, y = self.env.position
        if self.view == "absolute":
            out = np.zeros(ABSOLUTE_SIZE, dtype=np.float32)
            out[x] = out[MAX_SIZE + y] = 1.0
            out[-2], out[-1] = float(self.env.key), float(self.env.coin)
            return out
        out = self.planes[:, y:y + WINDOW, x:x + WINDOW].copy()
        if self.env.key:
            out[KEY] = 0.0
            out[HAS_KEY] = 1.0
        if self.env.coin:
            out[COIN] = 0.0
        return out

    def reset(self) -> np.ndarray:
        self.env.reset()
        return self.observe()

    def step(self, action: int):
        _, reward, terminated, truncated, info = self.env.step(int(action))
        return self.observe(), reward, terminated, truncated, info


class LayoutPool(gym.Env):
    """Gymnasium environment drawing one layout from a fixed pool at each episode start."""

    metadata = {"render_modes": []}

    def __init__(self, layouts: Sequence[Layout], view: str = "egocentric",
                 reward_mode: str = "sparse"):
        if not layouts:
            raise ValueError("a layout pool cannot be empty")
        super().__init__()
        self.layouts, self.view, self.reward_mode = list(layouts), view, reward_mode
        self.games: dict[int, ViewedGridQuest] = {}
        self.action_space = spaces.Discrete(4)
        shape = EGOCENTRIC_SHAPE if view == "egocentric" else (ABSOLUTE_SIZE,)
        self.observation_space = spaces.Box(0.0, 1.0, shape, dtype=np.float32)
        self.rng = random.Random(0)
        self.game = self._game(0)

    def _game(self, index: int) -> ViewedGridQuest:
        if index not in self.games:
            self.games[index] = ViewedGridQuest(self.layouts[index], self.view, self.reward_mode)
        return self.games[index]

    def reset(self, *, seed: int | None = None, options: dict | None = None):
        super().reset(seed=seed)
        if seed is not None:
            self.rng = random.Random(seed)
        self.game = self._game(self.rng.randrange(len(self.layouts)))
        return self.game.reset(), {"layout": self.game.layout}

    def step(self, action):
        return self.game.step(int(action))


def shortest_solution(layout: Layout) -> int | None:
    """Fewest moves from the start to the goal holding the key, under the game's own door rule."""
    start = (layout.start, layout.start == layout.key_at)
    queue = deque([(start, 0)])
    seen = {start}
    while queue:
        (cell, has_key), steps = queue.popleft()
        if cell == layout.goal_at and has_key:
            return steps
        for dx, dy in DIRECTIONS:
            nxt = (cell[0] + dx, cell[1] + dy)
            if not (0 <= nxt[0] < layout.size and 0 <= nxt[1] < layout.size):
                continue
            if nxt in layout.walls or (nxt == layout.door_at and not has_key):
                continue
            state = (nxt, has_key or nxt == layout.key_at)
            if state not in seen:
                seen.add(state)
                queue.append((state, steps + 1))
    return None


def path_score(success: bool, shortest: int, taken: int) -> float:
    """Success weighted by path length for one episode: shortest / max(taken, shortest), or 0."""
    return shortest / max(taken, shortest) if success else 0.0
