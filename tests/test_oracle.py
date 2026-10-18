"""Exercise the public movement/score contract on real GridQuest transitions."""

import random

from envs.gridquest import GridQuest
from playtest.oracle import GameContract, observe_transition


def contract_for(env):
    return GameContract(env.size, frozenset(env.wall), env.door_at, env.goal_at)


def move(env, contract, action):
    before_position, before_key = env.position, env.key
    _, _, _, _, info = env.step(action)
    return observe_transition(contract, before_position, before_key, action, info)


def test_clean_movement_ignores_an_injected_label():
    env = GridQuest()
    contract = contract_for(env)
    before_position, before_key = env.position, env.key
    _, _, _, _, info = env.step(3)
    info["bug_detected"] = "wall_gap"
    assert observe_transition(contract, before_position, before_key, 3, info) is None


def test_wall_gap_and_score_overflow_are_observable_without_bug_label():
    wall = GridQuest(bug="wall_gap")
    wall_contract = contract_for(wall)
    assert move(wall, wall_contract, 1) is None
    assert move(wall, wall_contract, 1) is None
    assert move(wall, wall_contract, 2) == "out_of_bounds"

    score = GridQuest(bug="score_overflow")
    score_contract = contract_for(score)
    for _ in range(3):
        assert move(score, score_contract, 3) is None
    assert move(score, score_contract, 2) is None
    assert move(score, score_contract, 3) == "score_out_of_bounds"


def test_blocked_goal_and_softlock_are_observable_without_bug_label():
    blocked = GridQuest(bug="unreachable_checkpoint")
    blocked_contract = contract_for(blocked)
    for _ in range(5):
        assert move(blocked, blocked_contract, 1) is None
    for _ in range(4):
        assert move(blocked, blocked_contract, 3) is None
    assert move(blocked, blocked_contract, 3) == "goal_entry_blocked"

    stuck = GridQuest(bug="softlock")
    stuck_contract = contract_for(stuck)
    for _ in range(4):
        assert move(stuck, stuck_contract, 3) is None
    assert move(stuck, stuck_contract, 1) == "legal_move_blocked"


def test_clean_seeded_play_remains_unflagged():
    for seed in range(5):
        env = GridQuest()
        contract = contract_for(env)
        env.reset(seed=seed)
        rng = random.Random(seed)
        for tick in range(300):
            before_position, before_key = env.position, env.key
            action = rng.randrange(4)
            _, _, terminated, truncated, info = env.step(action)
            assert observe_transition(contract, before_position, before_key, action, info) is None
            if terminated or truncated:
                env.reset(seed=seed * 100000 + tick)
