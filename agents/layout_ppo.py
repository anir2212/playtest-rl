"""Clipped PPO over GridQuest layout pools, kept apart from the frozen Slicer PPO.

``agents/ppo.py`` is pinned by earlier protocols and hard-wired to Slicer's six numbers, so this
module reuses only its GAE function. It adds a small convolutional network for the egocentric view,
a small MLP for the absolute control, and batched evaluation that records one result per layout.
"""
from __future__ import annotations

import hashlib
import random
from collections.abc import Sequence

import numpy as np
import torch
from torch import nn
from torch.distributions import Categorical

from agents.ppo import gae_advantages
from envs.gridquest import Layout
from envs.gridquest_views import (ABSOLUTE_SIZE, EGOCENTRIC_SHAPE, LayoutPool, ViewedGridQuest,
                                  path_score, shortest_solution)


class EgocentricActorCritic(nn.Module):
    def __init__(self):
        super().__init__()
        planes, side, _ = EGOCENTRIC_SHAPE
        after = (side - 3) // 2 + 1 - 2
        self.body = nn.Sequential(
            nn.Conv2d(planes, 16, 3, stride=2), nn.ReLU(), nn.Conv2d(16, 32, 3), nn.ReLU(),
            nn.Flatten(), nn.Linear(32 * after * after, 128), nn.ReLU())
        self.actor = nn.Linear(128, 4)
        self.critic = nn.Linear(128, 1)

    def forward(self, obs):
        body = self.body(obs)
        return Categorical(logits=self.actor(body)), self.critic(body).squeeze(-1)


class AbsoluteActorCritic(nn.Module):
    def __init__(self):
        super().__init__()
        self.body = nn.Sequential(nn.Linear(ABSOLUTE_SIZE, 128), nn.ReLU(), nn.Linear(128, 128),
                                  nn.ReLU())
        self.actor = nn.Linear(128, 4)
        self.critic = nn.Linear(128, 1)

    def forward(self, obs):
        body = self.body(obs)
        return Categorical(logits=self.actor(body)), self.critic(body).squeeze(-1)


def build_model(view: str) -> nn.Module:
    return EgocentricActorCritic() if view == "egocentric" else AbsoluteActorCritic()


def fingerprint(model: nn.Module) -> str:
    digest = hashlib.sha256()
    for name, value in sorted(model.state_dict().items()):
        digest.update(name.encode())
        digest.update(value.detach().cpu().numpy().tobytes())
    return digest.hexdigest()[:16]


def train(layouts: Sequence[Layout], seed: int, steps: int, view: str = "egocentric",
          reward_mode: str = "shaped", lr: float = 3e-4, ent_coef: float = 0.01,
          n_envs: int = 16, rollout: int = 128, epochs: int = 4, minibatch: int = 256,
          clip: float = 0.2, lam: float = 0.95, gamma: float = 0.99):
    """Train on ``layouts`` for ``steps`` environment steps; each episode draws one uniformly."""
    if steps < n_envs * rollout:
        raise ValueError("steps smaller than one rollout")
    torch.manual_seed(seed)
    order = np.random.default_rng(seed)
    envs = [LayoutPool(layouts, view, reward_mode) for _ in range(n_envs)]
    obs = np.stack([env.reset(seed=seed * 1000 + index)[0] for index, env in enumerate(envs)])
    model = build_model(view)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    consumed, history = 0, []
    while consumed + n_envs * rollout <= steps:
        states, actions, logprobs, rewards, dones, values = [], [], [], [], [], []
        finished = successes = 0
        for _ in range(rollout):
            with torch.no_grad():
                distribution, value = model(torch.as_tensor(obs))
                action = distribution.sample()
                logprob = distribution.log_prob(action)
            step_rewards, step_dones, next_obs = [], [], []
            for env, chosen in zip(envs, action.tolist()):
                after, reward, terminated, truncated, info = env.step(chosen)
                if terminated or truncated:
                    finished += 1
                    successes += int(info["success"])
                    after = env.reset()[0]
                next_obs.append(after)
                step_rewards.append(reward)
                step_dones.append(float(terminated or truncated))
            states.append(obs)
            actions.append(action.numpy())
            logprobs.append(logprob.numpy())
            rewards.append(step_rewards)
            dones.append(step_dones)
            values.append(value.numpy())
            obs = np.stack(next_obs)
        with torch.no_grad():
            _, last_value = model(torch.as_tensor(obs))
        rewards_np = np.asarray(rewards, dtype=np.float32)
        values_np = np.asarray(values, dtype=np.float32)
        advantages, returns = gae_advantages(rewards_np, values_np,
                                             np.asarray(dones, dtype=np.float32),
                                             last_value.numpy(), gamma=gamma, lam=lam)
        flat_states = torch.as_tensor(np.concatenate(states))
        flat_actions = torch.as_tensor(np.concatenate(actions), dtype=torch.int64)
        flat_logprobs = torch.as_tensor(np.concatenate(logprobs))
        flat_advantages = torch.as_tensor(advantages.reshape(-1))
        flat_returns = torch.as_tensor(returns.reshape(-1), dtype=torch.float32)
        flat_advantages = (flat_advantages - flat_advantages.mean()) / (flat_advantages.std() + 1e-8)
        size = len(flat_actions)
        for _ in range(epochs):
            for indices in np.array_split(order.permutation(size), max(1, size // minibatch)):
                distribution, value = model(flat_states[indices])
                ratio = torch.exp(distribution.log_prob(flat_actions[indices]) - flat_logprobs[indices])
                gain = flat_advantages[indices]
                policy_loss = -torch.min(ratio * gain, ratio.clamp(1 - clip, 1 + clip) * gain).mean()
                value_loss = 0.5 * (value - flat_returns[indices]).pow(2).mean()
                loss = policy_loss + 0.5 * value_loss - ent_coef * distribution.entropy().mean()
                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), 0.5)
                optimizer.step()
        consumed += n_envs * rollout
        history.append({"steps": consumed, "episodes": finished, "successes": successes})
    return model, {"steps": consumed, "history": history}


def evaluate(model: nn.Module | None, layouts: Sequence[Layout], seed: int, episodes: int,
             view: str = "egocentric", mode: str = "sampled", reward_mode: str = "shaped") -> dict:
    """Play every layout ``episodes`` times; returns per-layout mean success and path score.

    ``model=None`` is the uniform-random reference. ``mode`` is ``sampled`` (draw from the policy)
    or ``greedy`` (most likely action). Episodes run in one batch, so a seed fixes the whole set.
    """
    if mode not in ("sampled", "greedy") or (model is None and mode == "greedy"):
        raise ValueError("unknown evaluation mode")
    generator = torch.Generator().manual_seed(seed)
    uniform = random.Random(seed)
    shortest = [shortest_solution(layout) for layout in layouts]
    if None in shortest:
        raise ValueError("an unsolvable layout reached evaluation")
    games = [(index, ViewedGridQuest(layout, view, reward_mode))
             for index, layout in enumerate(layouts) for _ in range(episodes)]
    success = np.zeros(len(layouts), dtype=np.float64)
    score = np.zeros(len(layouts), dtype=np.float64)
    taken = np.zeros(len(layouts), dtype=np.float64)
    active = [(index, game, game.reset()) for index, game in games]
    while active:
        if model is None:
            chosen = [uniform.randrange(4) for _ in active]
        else:
            with torch.no_grad():
                distribution, _ = model(torch.as_tensor(np.stack([item[2] for item in active])))
            if mode == "greedy":
                chosen = distribution.probs.argmax(dim=1).tolist()
            else:
                chosen = torch.multinomial(distribution.probs, 1, generator=generator)[:, 0].tolist()
        remaining = []
        for (index, game, _), action in zip(active, chosen):
            obs, _, terminated, truncated, info = game.step(action)
            if terminated or truncated:
                won = bool(info["success"])
                success[index] += won
                score[index] += path_score(won, shortest[index], game.env.steps)
                taken[index] += game.env.steps
            else:
                remaining.append((index, game, obs))
        active = remaining
    return {"success": (success / episodes).tolist(), "spl": (score / episodes).tolist(),
            "mean_steps": (taken / episodes).tolist(), "shortest": shortest}
