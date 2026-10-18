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
