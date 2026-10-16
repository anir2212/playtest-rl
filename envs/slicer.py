"""Headless vector-observation arcade slicer for compact RL experiments."""
from __future__ import annotations

import numpy as np
import gymnasium as gym
from gymnasium import spaces


class Slicer(gym.Env):
    metadata = {"render_modes": ["ansi"]}

    def __init__(self, bomb_ratio: float = 0.2, horizon: int = 80,
                 render_mode: str | None = None):
        if not 0 <= bomb_ratio <= 1 or horizon < 10:
            raise ValueError("invalid difficulty")
        super().__init__()
        self.bomb_ratio = bomb_ratio
        self.horizon = horizon
        self.render_mode = render_mode
        self.action_space = spaces.Discrete(4)  # stay, left, right, slice
        self.observation_space = spaces.Box(low=-2, high=2, shape=(6,), dtype=np.float32)
        self.reset()

    def _spawn(self):
        self.object_x = float(self.np_random.uniform(0.05, 0.95))
        self.object_y = 0.0
        self.velocity_x = float(self.np_random.uniform(-0.015, 0.015))
        self.velocity_y = float(self.np_random.uniform(0.03, 0.055))
        self.is_bomb = bool(self.np_random.random() < self.bomb_ratio)

    def _obs(self):
        return np.array([self.cursor, self.object_x, self.object_y,
                         self.velocity_x, self.velocity_y, float(self.is_bomb)], dtype=np.float32)

    def reset(self, *, seed: int | None = None, options: dict | None = None):
        super().reset(seed=seed)
        self.cursor = 0.5
        self.ticks = 0
        self.score = 0
        self.sliced = 0
        self.bomb_hits = 0
        self._spawn()
        return self._obs(), {"score": 0, "sliced": 0, "bomb_hits": 0}

    def step(self, action: int):
        if not self.action_space.contains(action):
            raise ValueError("action outside action space")
        self.ticks += 1
        if action == 1:
            self.cursor = max(0.0, self.cursor - 0.12)
        elif action == 2:
            self.cursor = min(1.0, self.cursor + 0.12)
        self.object_x += self.velocity_x
        self.object_y += self.velocity_y
        reward = -0.005
        if action == 3 and abs(self.cursor - self.object_x) < 0.14 and self.object_y > 0.2:
            if self.is_bomb:
                reward = -1.0
                self.bomb_hits += 1
            else:
                reward = 1.0
                self.score += 1
                self.sliced += 1
            self._spawn()
        elif self.object_y >= 1.0:
            if not self.is_bomb:
                reward = -0.1
            self._spawn()
        return self._obs(), float(reward), False, bool(self.ticks >= self.horizon), {
            "score": self.score, "sliced": self.sliced, "bomb_hits": self.bomb_hits}

    def render(self):
        return f"tick={self.ticks} cursor={self.cursor:.2f} object={self.object_x:.2f},{self.object_y:.2f} bomb={self.is_bomb} score={self.score}"
