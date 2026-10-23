"""Detect contract violations from observed transitions, without injected bug labels."""

from __future__ import annotations

from dataclasses import dataclass

from envs.gridquest import DIRECTIONS


@dataclass(frozen=True)
class GameContract:
    size: int
    walls: frozenset[tuple[int, int]]
    door_at: tuple[int, int]
    goal_at: tuple[int, int]
    max_score: int = 11  # One coin point and ten goal points in GridQuest.


def observe_transition(
    contract: GameContract,
    before_position: tuple[int, int],
    before_key: bool,
    action: int,
    after: dict,
) -> str | None:
    """Return the first broken public game invariant, or None for a valid move."""
    after_position = tuple(after["position"])
    x, y = after_position
    if not (0 <= x < contract.size and 0 <= y < contract.size):
        return "out_of_bounds"
    if not (0 <= after["score"] <= contract.max_score):
        return "score_out_of_bounds"

    dx, dy = DIRECTIONS[action]
    intended = (before_position[0] + dx, before_position[1] + dy)
    legal = (
        0 <= intended[0] < contract.size
        and 0 <= intended[1] < contract.size
        and intended not in contract.walls
        and (intended != contract.door_at or before_key)
    )
    if legal and after_position != intended:
        return "goal_entry_blocked" if intended == contract.goal_at else "legal_move_blocked"
    if not legal and after_position != before_position:
        return "illegal_move"
    return None


# ------------------------------------------------------------------ contract v2 (23 October 2026)
# ``observe_transition`` above is the frozen v1 contract used by every earlier protocol. The
# v2 contract below predicts the whole public state after a move from the map's declared rules
# and compares it with what the game reports. It also never reads ``info["bug_detected"]``.

PUBLIC_FIELDS = ("position", "key", "coin", "score")


@dataclass(frozen=True)
class GameRules:
    """Declared GridQuest rules: map cells plus the scoring table."""
    size: int
    walls: frozenset[tuple[int, int]]
    door_at: tuple[int, int]
    goal_at: tuple[int, int]
    key_at: tuple[int, int]
    coin_at: tuple[int, int]
    coin_points: int = 1
    goal_points: int = 10

    @classmethod
    def from_layout(cls, layout) -> GameRules:
        return cls(layout.size, frozenset(layout.walls), layout.door_at, layout.goal_at,
                   layout.key_at, layout.coin_at)


def public_state(info: dict) -> dict:
    """The observable fields the v2 contract tracks, copied from a reset or step ``info``."""
    return {"position": tuple(info["position"]), "key": bool(info["key"]),
            "coin": bool(info["coin"]), "score": info["score"]}


def expected_transition(rules: GameRules, before: dict, action: int) -> dict:
    """Next public state under the declared rules, from the previous public state."""
    x, y = before["position"]
    dx, dy = DIRECTIONS[action]
    intended = (x + dx, y + dy)
    legal = (0 <= intended[0] < rules.size and 0 <= intended[1] < rules.size
             and intended not in rules.walls
             and (intended != rules.door_at or before["key"]))
    position = intended if legal else (x, y)
    key = before["key"] or position == rules.key_at
    coin, score = before["coin"], before["score"]
    if position == rules.coin_at and not coin:
        coin, score = True, score + rules.coin_points
    if position == rules.goal_at and key:
        score += rules.goal_points
    return {"position": position, "key": key, "coin": coin, "score": score}


def check_transition(rules: GameRules, before: dict, action: int, after: dict) -> str | None:
    """Return '<field>_mismatch' for the first public field that breaks the rules, else None."""
    expected = expected_transition(rules, before, action)
    observed = public_state(after)
    for field in PUBLIC_FIELDS:
        if observed[field] != expected[field]:
            return f"{field}_mismatch"
    return None
