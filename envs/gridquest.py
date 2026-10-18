"""Small discrete game with key, door, coin, hazard and injected defects."""
from __future__ import annotations

from collections import deque
import random

import gymnasium as gym
from gymnasium import spaces

DIRECTIONS = ((0, -1), (0, 1), (-1, 0), (1, 0))


class GridQuest(gym.Env):
    metadata = {"render_modes": ["ansi"]}

    def __init__(self, size: int = 6, difficulty: int = 0, bug: str | None = None,
                 reward_mode: str = "sparse", render_mode: str | None = None):
        if size < 6 or difficulty not in (0, 1, 2):
            raise ValueError("size >= 6 and difficulty in 0..2 required")
        if bug not in (None, "wall_gap", "score_overflow", "unreachable_checkpoint", "softlock"):
            raise ValueError("unknown bug")
        if reward_mode not in ("sparse", "shaped", "hackable", "fixed"):
            raise ValueError("unknown reward mode")
        super().__init__()
        self.size, self.difficulty, self.bug, self.reward_mode = size, difficulty, bug, reward_mode
        self.render_mode = render_mode
        self.action_space = spaces.Discrete(4)
        self.observation_space = spaces.Discrete((size + 2) ** 2 * 4)
        self.max_steps = size * size * (3 - difficulty // 2)
        self.key_at = (0, size - 1)
        self.coin_at = (size // 2, 0)
        self.door_at = (size - 2, size - 1)
        self.goal_at = (size - 1, size - 1)
        self.hazards = {(size - 2, 1), (2, size - 2)} if difficulty else {(size - 2, 1)}
        self.wall = {(2, 2), (2, 3)}
        self.reset()

    def _obs(self) -> int:
        x, y = self.position
        return (((y + 1) * (self.size + 2) + x + 1) * 2 + int(self.key)) * 2 + int(self.coin)

    def _info(self, bug_detected: str | None = None) -> dict:
        return {"success": self.position == self.goal_at and self.key,
                "position": self.position, "key": self.key, "coin": self.coin,
                "bug_detected": bug_detected, "score": self.score}

    def reset(self, *, seed: int | None = None, options: dict | None = None):
        super().reset(seed=seed)
        self.position = (0, 0)
        self.key = False
        self.coin = False
        self.steps = 0
        self.score = 0
        self.stuck = False
        return self._obs(), self._info()

    def step(self, action: int):
        if not self.action_space.contains(action):
            raise ValueError("action outside action space")
        self.steps += 1
        x, y = self.position
        had_key = self.key
        had_coin = self.coin
        dx, dy = DIRECTIONS[int(action)]
        nx, ny = x + dx, y + dy
        bug_detected = None
        if self.stuck:
            bug_detected = "softlock"
        elif self.bug == "wall_gap" and (nx, ny) == (-1, 2):
            self.position = (nx, ny)
            bug_detected = "wall_gap"
        elif (0 <= nx < self.size and 0 <= ny < self.size and
              (nx, ny) not in self.wall and
              (nx, ny) != self.door_at or
              (nx, ny) == self.door_at and self.key) and not (
                  self.bug == "unreachable_checkpoint" and (nx, ny) == self.goal_at):
            self.position = (nx, ny)
        if self.position == self.key_at:
            self.key = True
        if self.position == self.coin_at and not self.coin:
            self.coin = True
            self.score += 1
        if self.bug == "softlock" and self.position == (self.size - 2, 0):
            self.stuck = True
        if self.bug == "score_overflow" and had_coin and self.position == self.coin_at:
            self.score = 1_000_000
            bug_detected = "score_overflow"
        success = self.position == self.goal_at and self.key
        if success:
            self.score += 10
        if self.reward_mode == "sparse":
            reward = 1.0 if success else -0.01
        elif self.reward_mode == "shaped":
            reward = (1.0 if success else -0.01) + (0.2 if self.key and not had_key else 0.0)
        elif self.reward_mode == "hackable":
            reward = (1.0 if success else -0.01) + (0.2 if self.position == self.key_at and not (x, y) == self.key_at else 0.0)
        else:
            reward = (1.0 if success else -0.01) + (0.2 if self.key and not had_key else 0.0)
        if self.position in self.hazards:
            reward -= 0.05
        if self.bug == "unreachable_checkpoint" and self.steps == self.max_steps and not success:
            bug_detected = "unreachable_checkpoint" if not self._reachable_goal() else None
        return self._obs(), float(reward), bool(success or bug_detected in ("wall_gap", "score_overflow")), bool(self.steps >= self.max_steps), self._info(bug_detected)

    def _reachable_goal(self) -> bool:
        if self.bug == "unreachable_checkpoint":
            return False
        queue = deque([(0, 0, False)])
        seen = set(queue)
        while queue:
            x, y, key = queue.popleft()
            if (x, y) == self.goal_at and key:
                return True
            for dx, dy in DIRECTIONS:
                nx, ny = x + dx, y + dy
                if not (0 <= nx < self.size and 0 <= ny < self.size) or (nx, ny) in self.wall:
                    continue
                nk = key or (nx, ny) == self.key_at
                if (nx, ny) == self.door_at and not nk:
                    continue
                nxt = (nx, ny, nk)
                if nxt not in seen:
                    seen.add(nxt)
                    queue.append(nxt)
        return False

    def render(self):
        cells = []
        for y in range(self.size):
            cells.append("".join("@" if (x, y) == self.position else
                                 "#" if (x, y) in self.wall else
                                 "K" if (x, y) == self.key_at and not self.key else
                                 "$" if (x, y) == self.coin_at and not self.coin else
                                 "D" if (x, y) == self.door_at else
                                 "G" if (x, y) == self.goal_at else "."
                                 for x in range(self.size)))
        return "\n".join(cells)
