"""Seeded automated playtest comparison on four injected defects."""
from __future__ import annotations

from collections import defaultdict, deque
import math
import random

from envs.gridquest import DIRECTIONS, GridQuest, Layout
from playtest.oracle import (GameContract, GameRules, check_transition, observe_transition,
                             public_state)

BUGS = ("wall_gap", "score_overflow", "unreachable_checkpoint", "softlock")
POLICIES = ("random", "scripted", "coverage", "q_count")
# Defects added after the frozen protocols; kept out of BUGS so earlier runs are unchanged.
NEW_DEFECTS = ("coin_not_scored",)
ORACLES = ("v1", "v2")


def _scripted_action(env: GridQuest):
    """Generic key-then-goal path; has no bug-location knowledge."""
    target = env.key_at if not env.key else env.goal_at
    queue = deque([(env.position, [])])
    seen = {env.position}
    while queue:
        pos, path = queue.popleft()
        if pos == target:
            return path[0] if path else 0
        for action, (dx, dy) in enumerate(DIRECTIONS):
            nxt = (pos[0] + dx, pos[1] + dy)
            if not (0 <= nxt[0] < env.size and 0 <= nxt[1] < env.size):
                continue
            if nxt in env.wall or (nxt == env.door_at and not env.key) or nxt in seen:
                continue
            seen.add(nxt)
            queue.append((nxt, path + [action]))
    return 0


def hunt(bug: str | None, policy: str, seed: int, max_steps: int = 1200,
         layout: Layout | None = None, oracle: str = "v1"):
    """Search for a contract violation; ``bug=None`` measures false alarms on a clean map.

    ``oracle="v1"`` is the frozen movement/score-range contract used by every earlier
    protocol; ``"v2"`` checks the whole public state against the declared rules.
    """
    if (bug is not None and bug not in BUGS + NEW_DEFECTS) or policy not in POLICIES:
        raise ValueError("unknown bug or policy")
    if oracle not in ORACLES:
        raise ValueError("unknown oracle")
    rng = random.Random(seed)
    env = GridQuest(bug=bug, layout=layout)
    contract = GameContract(env.size, frozenset(env.wall), env.door_at, env.goal_at)
    rules = GameRules.from_layout(env.layout)
    state, info = env.reset(seed=seed)
    public = public_state(info)
    seen_positions = {env.position}
    visits = defaultdict(int)
    visits[env.position] = 1
    tried = defaultdict(int)
    q = defaultdict(lambda: [0.0] * 4)
    for tick in range(1, max_steps + 1):
        if policy == "random":
            action = rng.randrange(4)
        elif policy == "scripted":
            action = _scripted_action(env)
        elif policy == "coverage":
            action = min(range(4), key=lambda a: (tried[(state, a)],
                visits[(env.position[0] + DIRECTIONS[a][0], env.position[1] + DIRECTIONS[a][1])], rng.random()))
        else:
            action = rng.randrange(4) if rng.random() < 0.2 else max(range(4), key=lambda a: q[state][a])
        tried[(state, action)] += 1
        before_position, before_key = env.position, env.key
        next_state, _, terminated, truncated, info = env.step(action)
        visits[env.position] += 1
        seen_positions.add(env.position)
        if policy == "q_count":
            intrinsic = 1 / math.sqrt(visits[env.position])
            q[state][action] += 0.3 * (intrinsic + (0 if terminated or truncated else 0.95 * max(q[next_state])) - q[state][action])
        state = next_state
        violation = (observe_transition(contract, before_position, before_key, action, info)
                     if oracle == "v1" else check_transition(rules, public, action, info))
        public = public_state(info)
        if violation:
            return {"time_to_discovery": tick, "detected": violation,
                    "position_coverage": len(seen_positions) / env.size ** 2,
                    "position_visits": [[visits[(x, y)] for x in range(env.size)]
                                        for y in range(env.size)]}
        if terminated or truncated:
            state, info = env.reset(seed=seed * 100000 + tick)
            public = public_state(info)
            seen_positions.add(env.position)
    return {"time_to_discovery": None, "detected": None,
            "position_coverage": len(seen_positions) / env.size ** 2,
            "position_visits": [[visits[(x, y)] for x in range(env.size)]
                                for y in range(env.size)]}
