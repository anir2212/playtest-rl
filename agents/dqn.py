"""Small Double DQN with replay, Huber loss and a periodically copied target."""
from __future__ import annotations

from collections import deque
import random
import numpy as np
import torch
from torch import nn

from envs.slicer import Slicer


class ReplayBuffer:
    def __init__(self, capacity: int):
        self.data = deque(maxlen=capacity)

    def add(self, obs, action, reward, next_obs, done):
        self.data.append((np.array(obs, copy=True), int(action), float(reward),
                          np.array(next_obs, copy=True), bool(done)))

    def sample(self, batch_size: int, rng: random.Random):
        if batch_size > len(self.data):
            raise ValueError("batch exceeds replay")
        rows = rng.sample(list(self.data), batch_size)
        obs, actions, rewards, next_obs, dones = zip(*rows)
        return (torch.as_tensor(np.stack(obs), dtype=torch.float32),
                torch.as_tensor(actions, dtype=torch.int64),
                torch.as_tensor(rewards, dtype=torch.float32),
                torch.as_tensor(np.stack(next_obs), dtype=torch.float32),
                torch.as_tensor(dones, dtype=torch.float32))


class QNetwork(nn.Module):
    def __init__(self, observations: int = 6, actions: int = 4):
        super().__init__()
        self.layers = nn.Sequential(nn.Linear(observations, 64), nn.ReLU(),
                                    nn.Linear(64, 64), nn.ReLU(), nn.Linear(64, actions))

    def forward(self, obs):
        return self.layers(obs)


def double_dqn_target(online: QNetwork, target: QNetwork, next_obs, rewards, dones,
                      gamma: float):
    with torch.no_grad():
        best_actions = online(next_obs).argmax(dim=1, keepdim=True)
        next_values = target(next_obs).gather(1, best_actions).squeeze(1)
        return rewards + gamma * (1 - dones) * next_values


def train(seed: int, steps: int = 5000, double: bool = True,
          bomb_ratio: float = 0.2, batch_size: int = 64):
    torch.set_num_threads(2)
    torch.manual_seed(seed)
    rng = random.Random(seed)
    env = Slicer(bomb_ratio=bomb_ratio)
    obs, _ = env.reset(seed=seed)
    online, target = QNetwork(), QNetwork()
    target.load_state_dict(online.state_dict())
    optimizer = torch.optim.Adam(online.parameters(), lr=5e-4)
    replay = ReplayBuffer(10000)
    returns, episode_return, losses = [], 0.0, []
    for step in range(steps):
        epsilon = max(0.05, 1.0 - 0.95 * step / max(1, int(steps * 0.7)))
        if rng.random() < epsilon:
            action = rng.randrange(4)
        else:
            with torch.no_grad():
                action = int(online(torch.as_tensor(obs, dtype=torch.float32)).argmax())
        next_obs, reward, terminated, truncated, _ = env.step(action)
        done = terminated or truncated
        replay.add(obs, action, reward, next_obs, done)
        episode_return += reward
        obs = next_obs
        if len(replay.data) >= batch_size and step % 4 == 0:
            states, actions, rewards, next_states, dones = replay.sample(batch_size, rng)
            if double:
                targets = double_dqn_target(online, target, next_states, rewards, dones, 0.99)
            else:
                with torch.no_grad():
                    targets = rewards + 0.99 * (1 - dones) * target(next_states).max(dim=1).values
            values = online(states).gather(1, actions[:, None]).squeeze(1)
            loss = nn.functional.smooth_l1_loss(values, targets)
            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(online.parameters(), 10)
            optimizer.step()
            losses.append(float(loss.detach()))
        if (step + 1) % 250 == 0:
            target.load_state_dict(online.state_dict())
        if done:
            returns.append(episode_return)
            episode_return = 0.0
            obs, _ = env.reset(seed=seed * 100000 + step)
    env.close()
    return online, {"train_returns": returns, "mean_loss": float(np.mean(losses)) if losses else None,
                    "steps": steps, "double": double}


def evaluate(model: QNetwork, seed: int, episodes: int = 30, bomb_ratio: float = 0.2):
    env = Slicer(bomb_ratio=bomb_ratio)
    scores, bombs, returns = [], [], []
    for episode in range(episodes):
        obs, _ = env.reset(seed=seed * 100000 + episode)
        total = 0.0
        for _ in range(env.horizon):
            with torch.no_grad():
                action = int(model(torch.as_tensor(obs, dtype=torch.float32)).argmax())
            obs, reward, terminated, truncated, info = env.step(action)
            total += reward
            if terminated or truncated:
                break
        scores.append(info["score"])
        bombs.append(info["bomb_hits"])
        returns.append(total)
    env.close()
    return {"mean_score": float(np.mean(scores)), "mean_bomb_hits": float(np.mean(bombs)),
            "mean_return": float(np.mean(returns))}
