"""New defect (coin_not_scored), the v2 rule-model oracle and the new-defect protocol guards.

No test runs the predeclared hunt; the defect is driven through single real transitions.
"""

import json
import random
import shutil
from collections import deque

import pytest

from envs.gridquest import DIRECTIONS, GridQuest, default_layout
from playtest import new_defect
from playtest.bug_hunt import BUGS, hunt
from playtest.levels import generate
from playtest.oracle import GameContract, GameRules, check_transition, observe_transition, public_state

MAPS = [default_layout(), *generate(20261020, 8)]
ROOT = new_defect.ROOT


def path_to(layout, target):
    """Actions from the start to ``target`` without passing the door (no key needed)."""
    queue, seen = deque([(layout.start, [])]), {layout.start}
    while queue:
        cell, actions = queue.popleft()
        if cell == target:
            return actions
        for action, (dx, dy) in enumerate(DIRECTIONS):
            nxt = (cell[0] + dx, cell[1] + dy)
            if (0 <= nxt[0] < layout.size and 0 <= nxt[1] < layout.size and nxt not in layout.walls
                    and nxt != layout.door_at and nxt not in seen):
                seen.add(nxt)
                queue.append((nxt, actions + [action]))
    raise AssertionError("target unreachable")


def both_oracles(env, before_info, action):
    rules = GameRules.from_layout(env.layout)
    contract = GameContract(env.size, frozenset(env.wall), env.door_at, env.goal_at)
    before = public_state(before_info)
    info = env.step(action)[4]
    return (observe_transition(contract, before["position"], before["key"], action, info),
            check_transition(rules, before, action, info), info)


@pytest.mark.parametrize("index", range(len(MAPS)))
def test_coin_defect_is_invisible_to_v1_and_caught_by_v2_on_every_map(index):
    layout = MAPS[index]
    clean, broken = GridQuest(layout=layout), GridQuest(bug="coin_not_scored", layout=layout)
    clean_info, broken_info = clean.reset(seed=0)[1], broken.reset(seed=0)[1]
    actions = path_to(layout, layout.coin_at)
    for step, action in enumerate(actions):
        v1_clean, v2_clean, clean_info = both_oracles(clean, clean_info, action)
        v1, v2, broken_info = both_oracles(broken, broken_info, action)
        assert v1_clean is None and v2_clean is None and v1 is None
        if step < len(actions) - 1:
            assert v2 is None
    assert broken_info["coin"] and broken_info["score"] == 0 and clean_info["score"] == 1
    assert v2 == "score_mismatch"


def test_v2_ignores_the_injected_label():
    layout = default_layout()
    env = GridQuest(bug="coin_not_scored")
    info = env.reset(seed=0)[1]
    rules = GameRules.from_layout(layout)
    for action in path_to(layout, layout.coin_at):
        before = public_state(info)
        info = env.step(action)[4]
    assert info["bug_detected"] == "coin_not_scored"
    tampered = {**info, "bug_detected": None}
    assert check_transition(rules, before, action, tampered) == "score_mismatch"
    clean = GridQuest()
    start = public_state(clean.reset(seed=0)[1])
    moved = {**clean.step(3)[4], "bug_detected": "coin_not_scored"}
    assert check_transition(rules, start, 3, moved) is None


def test_v2_raises_nothing_on_clean_random_play_on_every_map():
    for layout in MAPS:
        rules = GameRules.from_layout(layout)
        for seed in range(3):
            env = GridQuest(layout=layout)
            info = env.reset(seed=seed)[1]
            rng = random.Random(seed)
            for tick in range(400):
                action = rng.randrange(4)
                before = public_state(info)
                _, _, terminated, truncated, info = env.step(action)
                assert check_transition(rules, before, action, info) is None
                if terminated or truncated:
                    info = env.reset(seed=seed * 100000 + tick)[1]


def test_v2_also_flags_the_four_original_defects():
    expected = {"wall_gap": "position_mismatch", "score_overflow": "score_mismatch",
                "unreachable_checkpoint": "position_mismatch", "softlock": "position_mismatch"}
    for bug in BUGS:
        assert hunt(bug, "coverage", 0, oracle="v2")["detected"] == expected[bug]


def test_v1_hunts_still_reproduce_the_committed_original_map_rows():
    saved = json.loads((ROOT / "evidence" / "playtest_contract_oracle.json").read_text())
    for row in saved["rows"]:
        result = hunt(row["bug"], row["policy"], row["seed"], max_steps=1200)
        assert (result["time_to_discovery"], result["detected"]) == (row["time_to_discovery"], row["detected"])


def test_unknown_defect_or_oracle_is_rejected():
    with pytest.raises(ValueError):
        hunt("coin_lost", "random", 0)
    with pytest.raises(ValueError):
        hunt("coin_not_scored", "random", 0, oracle="v3")


def test_protocol_guards(tmp_path):
    new_defect.check_protocol(new_defect.PROTOCOL)
    for change in ({"hunt_seeds": [0, 1, 2, 3]}, {"hunt_seeds": [0, 0, 1, 2, 3]},
                   {"defect": "softlock"}, {"oracles": {"v1": "only"}}):
        with pytest.raises(ValueError):
            new_defect.check_protocol({**new_defect.PROTOCOL, **change})
    for rel in new_defect.FROZEN_INPUTS.values():
        (tmp_path / rel).parent.mkdir(parents=True, exist_ok=True)
        shutil.copy(ROOT / rel, tmp_path / rel)
    path = tmp_path / "protocol.json"
    new_defect.write_protocol(path, root=tmp_path)
    with pytest.raises(FileExistsError):
        new_defect.write_protocol(path, root=tmp_path)
    new_defect.load_protocol(path, root=tmp_path)
    assert len(new_defect.held_out_layouts(tmp_path)) == 8
    body = json.loads(path.read_text())
    body["max_steps"] = 5000
    path.write_text(json.dumps(body))
    with pytest.raises(RuntimeError):
        new_defect.load_protocol(path, root=tmp_path)
    frozen = json.loads((tmp_path / new_defect.FROZEN_INPUTS["unseen_protocol"]).read_text())
    frozen["levels"]["sha256"] = "0" * 64
    (tmp_path / new_defect.FROZEN_INPUTS["unseen_protocol"]).write_text(json.dumps(frozen))
    with pytest.raises(RuntimeError):
        new_defect.held_out_layouts(tmp_path)


def test_committed_protocol_file_matches_the_constant():
    saved = new_defect.load_protocol()
    assert saved["name"] == "gridquest-new-defect-v1"
