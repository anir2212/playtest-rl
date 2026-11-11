"""Contract v3: the frozen v2 public-state contract plus the declared step reward.

``playtest/oracle.py`` and ``playtest/bug_hunt.py`` stay byte-identical, because earlier
protocols pin their hashes. This module therefore adds a pure reward check on top of the
unchanged ``check_transition`` and its own copy of the hunt loop. ``hunt_v3`` selects actions,
stops and resets exactly as ``bug_hunt.hunt`` does; the only differences are the oracle call
and the recorded ``first_success_step``.

The reward table is transcribed from the clean rule in ``GridQuest.step``. The oracle predicts
the reward from the state the rules expect, so it reads neither ``info["success"]`` nor
``info["bug_detected"]``.
"""
from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
import math
import random

from envs.gridquest import DIRECTIONS, GridQuest, Layout
from playtest.bug_hunt import BUGS, NEW_DEFECTS, POLICIES, _scripted_action
from playtest.oracle import GameRules, check_transition, expected_transition, public_state

REWARD_MODES = ("sparse", "shaped", "hackable", "fixed")
REWARD_TOLERANCE = 1e-9  # Absorbs float addition order only; every declared amount is far larger.


@dataclass(frozen=True)
class RewardRules:
    """Declared GridQuest rules: the v2 state rules, the hazard cells and the reward table."""
    state: GameRules
    hazards: frozenset[tuple[int, int]]
    mode: str = "sparse"
    goal_reward: float = 1.0
    step_reward: float = -0.01
    key_bonus: float = 0.2
    hazard_penalty: float = -0.05

    @classmethod
    def from_layout(cls, layout: Layout, mode: str = "sparse") -> RewardRules:
        if mode not in REWARD_MODES:
            raise ValueError("unknown reward mode")
        return cls(GameRules.from_layout(layout), frozenset(layout.hazards), mode)


def expected_reward(rules: RewardRules, before: dict, expected_after: dict) -> float:
    """Reward the declared table pays for moving from ``before`` to ``expected_after``."""
    position = expected_after["position"]
    success = position == rules.state.goal_at and expected_after["key"]
    reward = rules.goal_reward if success else rules.step_reward
    if rules.mode in ("shaped", "fixed") and expected_after["key"] and not before["key"]:
        reward += rules.key_bonus
    if (rules.mode == "hackable" and position == rules.state.key_at
            and before["position"] != rules.state.key_at):
        reward += rules.key_bonus
    if position in rules.hazards:
        reward += rules.hazard_penalty
    return reward


def check_transition_v3(rules: RewardRules, before: dict, action: int, after: dict,
                        reward: float) -> str | None:
    """Return v2's violation if the state is wrong, else 'reward_mismatch' or None."""
    violation = check_transition(rules.state, before, action, after)
    if violation:
        return violation
    expected = expected_reward(rules, before, expected_transition(rules.state, before, action))
    if not math.isclose(float(reward), expected, abs_tol=REWARD_TOLERANCE):
        return "reward_mismatch"
    return None


def hunt_v3(bug: str | None, policy: str, seed: int, max_steps: int = 1200,
            layout: Layout | None = None) -> dict:
    """Search for a v3 violation under the frozen hunt's action, stop and reset rules.

    ``first_success_step`` is the first step whose reported ``info["success"]`` is true. It is a
    measurement of whether the hunter reached the goal with the key and never feeds the oracle.
    """
    if (bug is not None and bug not in BUGS + NEW_DEFECTS) or policy not in POLICIES:
        raise ValueError("unknown bug or policy")
    rng = random.Random(seed)
    env = GridQuest(bug=bug, layout=layout)
    rules = RewardRules.from_layout(env.layout, env.reward_mode)
    state, info = env.reset(seed=seed)
    public = public_state(info)
    seen_positions = {env.position}
    visits = defaultdict(int)
    visits[env.position] = 1
    tried = defaultdict(int)
    q = defaultdict(lambda: [0.0] * 4)
    first_success_step = None

    def outcome(tick: int | None, violation: str | None) -> dict:
        return {"time_to_discovery": tick, "detected": violation,
                "position_coverage": len(seen_positions) / env.size ** 2,
                "position_visits": [[visits[(x, y)] for x in range(env.size)]
                                    for y in range(env.size)],
                "first_success_step": first_success_step}

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
        next_state, reward, terminated, truncated, info = env.step(action)
        visits[env.position] += 1
        seen_positions.add(env.position)
        if policy == "q_count":
            intrinsic = 1 / math.sqrt(visits[env.position])
            q[state][action] += 0.3 * (intrinsic + (0 if terminated or truncated else 0.95 * max(q[next_state])) - q[state][action])
        state = next_state
        if first_success_step is None and info["success"]:
            first_success_step = tick
        violation = check_transition_v3(rules, public, action, info, reward)
        public = public_state(info)
        if violation:
            return outcome(tick, violation)
        if terminated or truncated:
            state, info = env.reset(seed=seed * 100000 + tick)
            public = public_state(info)
            seen_positions.add(env.position)
    return outcome(None, None)
