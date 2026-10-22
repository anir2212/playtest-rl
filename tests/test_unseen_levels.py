"""Held-out GridQuest maps: generation, validity, defect mechanics and protocol guards."""

import json
import random
from dataclasses import replace

import pytest

from envs.gridquest import DIRECTIONS, GridQuest, Layout, default_layout
from playtest import unseen
from playtest.levels import generate, manifest, validate
from playtest.oracle import GameContract, observe_transition

LEVELS = generate(20261020, 8)


def contract(env):
    return GameContract(env.size, frozenset(env.wall), env.door_at, env.goal_at)


def step(env, action):
    before = (env.position, env.key)
    info = env.step(action)[4]
    return observe_transition(contract(env), before[0], before[1], action, info)


def test_default_layout_is_the_original_map():
    assert validate(default_layout()) == []
    a, b = GridQuest(bug="softlock"), GridQuest(bug="softlock", layout=default_layout())
    rng = random.Random(5)
    for _ in range(300):
        action = rng.randrange(4)
        out_a, out_b = a.step(action), b.step(action)
        assert out_a[:4] == out_b[:4] and out_a[4] == out_b[4]


def test_generation_is_deterministic_valid_distinct_and_excludes_the_original():
    assert manifest(20261020, 8)["sha256"] == manifest(20261020, 8)["sha256"]
    assert len(set(LEVELS)) == 8
    assert {level.size for level in LEVELS} == {6, 7, 8}
    for level in LEVELS:
        assert validate(level) == []
        assert level != default_layout(level.size)
        assert Layout.from_json(json.loads(json.dumps(level.to_json()))) == level


def test_validator_rejects_an_unreachable_goal():
    level = LEVELS[0]
    gx, gy = level.goal_at
    ring = {(gx + dx, gy + dy) for dx, dy in DIRECTIONS} - {level.door_at}
    blocked = replace(level, walls=frozenset(level.walls | ring), door_at=level.door_at)
    door_ring = {(level.door_at[0] + dx, level.door_at[1] + dy) for dx, dy in DIRECTIONS}
    sealed = replace(blocked, walls=frozenset(blocked.walls | (door_ring - {level.goal_at})))
    assert validate(sealed)


@pytest.mark.parametrize("index", range(8))
def test_each_location_bound_defect_is_observable_on_every_map(index):
    level = LEVELS[index]
    gx, gy = level.wall_gap
    edge = next((gx + dx, gy + dy) for dx, dy in DIRECTIONS
                if 0 <= gx + dx < level.size and 0 <= gy + dy < level.size)
    env = GridQuest(bug="wall_gap", layout=level)
    env.position = edge
    action = DIRECTIONS.index((gx - edge[0], gy - edge[1]))
    assert step(env, action) == "out_of_bounds"

    trap = GridQuest(bug="softlock", layout=level)
    tx, ty = level.softlock_at
    neighbour, action = next(((tx - dx, ty - dy), DIRECTIONS.index((dx, dy)))
                             for dx, dy in DIRECTIONS
                             if 0 <= tx - dx < level.size and 0 <= ty - dy < level.size
                             and (tx - dx, ty - dy) not in level.walls
                             and (tx - dx, ty - dy) != level.door_at)
    trap.position = neighbour
    assert step(trap, action) is None and trap.position == level.softlock_at
    legal = next(a for a, (dx, dy) in enumerate(DIRECTIONS)
                 if 0 <= tx + dx < level.size and 0 <= ty + dy < level.size
                 and (tx + dx, ty + dy) not in level.walls and (tx + dx, ty + dy) != level.door_at)
    assert step(trap, legal) == "legal_move_blocked"


def test_clean_maps_raise_no_violation_on_a_short_random_walk():
    for level in LEVELS:
        env = GridQuest(layout=level)
        rng = random.Random(11)
        env.reset(seed=11)
        for _ in range(150):
            before, action = (env.position, env.key), rng.randrange(4)
            _, _, terminated, truncated, info = env.step(action)
            assert observe_transition(contract(env), *before, action, info) is None
            if terminated or truncated:  # As the hunt harness does.
                env.reset(seed=12)


def test_protocol_is_never_overwritten_and_run_refuses_changed_maps(tmp_path):
    path = tmp_path / "protocol.json"
    body = unseen.write_protocol(path)
    with pytest.raises(FileExistsError):
        unseen.write_protocol(path)
    body["levels"]["sha256"] = "0" * 64
    path.write_text(json.dumps(body))
    with pytest.raises(RuntimeError, match="differ"):
        unseen.run(path, tmp_path / "out.json")


def test_wilson_interval_bounds():
    assert unseen.wilson(0, 40)[0] == 0.0
    assert unseen.wilson(40, 40)[1] == 1.0
    low, high = unseen.wilson(20, 40)
    assert low < 0.5 < high
