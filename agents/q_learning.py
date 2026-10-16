"""Tabular Q learning and reproducible evaluation."""
from __future__ import annotations

from collections import defaultdict
import random
import numpy as np

from envs.gridquest import GridQuest


def train(seed: int, episodes: int = 800, alpha: float = 0.25,
          gamma: float = 0.95, reward_mode: str = "shaped",
          difficulty: int = 0, curiosity_bonus: float = 0.0):
    if episodes < 1 or not 0 < alpha <= 1 or not 0 <= gamma <= 1:
        raise ValueError("invalid training hyperparameters")
    rng = random.Random(seed)
    env = GridQuest(reward_mode=reward_mode, difficulty=difficulty)
    q = defaultdict(lambda: np.zeros(env.action_space.n, dtype=np.float64))
    visits = defaultdict(int)
    history = []
    for episode in range(episodes):
        state, _ = env.reset(seed=seed * 100000 + episode)
        epsilon = max(0.05, 0.9 * (1 - episode / episodes))
        total = 0.0
        for _ in range(env.max_steps):
            action = rng.randrange(env.action_space.n) if rng.random() < epsilon else int(np.argmax(q[state]))
            next_state, reward, terminated, truncated, info = env.step(action)
            visits[next_state] += 1
            if curiosity_bonus:
                reward += curiosity_bonus / np.sqrt(visits[next_state])
            target = reward + (0 if terminated or truncated else gamma * float(np.max(q[next_state])))
            q[state][action] += alpha * (target - q[state][action])
            state = next_state
            total += reward
            if terminated or truncated:
                break
        history.append({"episode": episode, "return": total, "success": bool(info["success"]), "steps": env.steps})
    return q, history


def evaluate(q, seed: int, episodes: int = 100, difficulty: int = 0):
    env = GridQuest(difficulty=difficulty)
    wins, total_steps, states = 0, 0, set()
    for episode in range(episodes):
        state, _ = env.reset(seed=seed * 100000 + episode)
        for _ in range(env.max_steps):
            states.add(state)
            state, _, terminated, truncated, info = env.step(int(np.argmax(q[state])))
            if terminated or truncated:
                wins += int(info["success"])
                total_steps += env.steps
                break
    return {"success_rate": wins / episodes, "mean_steps": total_steps / episodes,
            "visited_states": len(states)}
