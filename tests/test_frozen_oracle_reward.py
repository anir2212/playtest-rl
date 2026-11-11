"""The reward challenge must expose a real fault without changing the frozen oracle."""

import json
import shutil
from pathlib import Path

import pytest

from envs.gridquest import GridQuest, default_layout
from playtest.bug_hunt import _scripted_action
from playtest.frozen_oracle_reward import load_protocol, scripted_exposure
from playtest.oracle import GameRules, check_transition, public_state


def test_successful_goal_has_wrong_reward_but_unchanged_public_state() -> None:
    layout = default_layout()
    for bug, expected_reward in ((None, 1.0), ("goal_reward_missing", 0.0)):
        env = GridQuest(layout=layout, bug=bug)
        _, info = env.reset(seed=0)
        rules = GameRules.from_layout(layout)
        before = public_state(info)
        for _ in range(env.max_steps):
            action = _scripted_action(env)
            _, reward, terminated, truncated, info = env.step(action)
            assert check_transition(rules, before, action, info) is None
            before = public_state(info)
            if info["success"]:
                assert terminated and not truncated
                assert reward == expected_reward
                assert info["score"] == 10
                break
        else:
            pytest.fail("scripted path never reached the goal")

    clean = scripted_exposure(layout, 0, "goal_reward_missing")
    assert clean["reached_goal"] and clean["fault_exposed"]


def test_frozen_oracle_hash_guard_rejects_a_modified_contract(tmp_path: Path) -> None:
    root = Path(__file__).resolve().parents[1]
    protocol = json.loads((root / "evidence/frozen_oracle_reward_protocol.json").read_text())
    for rel in protocol["frozen_inputs_sha256"]:
        destination = tmp_path / rel
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(root / rel, destination)
    destination = tmp_path / "evidence/frozen_oracle_reward_protocol.json"
    shutil.copyfile(root / "evidence/frozen_oracle_reward_protocol.json", destination)
    assert load_protocol(tmp_path)["name"] == protocol["name"]
    (tmp_path / "playtest/oracle.py").write_text("changed")
    with pytest.raises(RuntimeError, match="playtest/oracle.py"):
        load_protocol(tmp_path)
