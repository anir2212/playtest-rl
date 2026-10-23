"""SB3 reference protocol: structure, guards, callback placement and a tiny end-to-end run.

No test trains or evaluates an SB3 model on a held-out setting or for the protocol
budget; the small runs use a test-only setting (bomb share 0.1) and tiny budgets.
"""

import copy
import json
import shutil

import pytest

from analysis import curves, difficulty, sb3_reference as sb3
from experiments.run import eval_sb3

TEST_SETTING = {"bomb_ratio": 0.1, "speed": 1.0}


def tiny_protocol(**changes):
    protocol = copy.deepcopy(sb3.PROTOCOL)
    protocol.update({"name": "test-only", "total_steps": 1024, "checkpoint_every": 512,
                     "dqn_epsilon_decay_steps": 512, "sb3_dqn_exploration_fraction": 0.5,
                     "eval_episodes": 2,
                     "settings": {"train": {"bomb_ratio": 0.2, "speed": 1.0},
                                  "test_only": dict(TEST_SETTING)},
                     "held_out": ["test_only"]})
    protocol.update(changes)
    return protocol


def test_protocol_matches_the_frozen_curve_and_difficulty_designs():
    protocol = sb3.PROTOCOL
    sb3.check_protocol(protocol)
    for key in ("training_seeds", "total_steps", "checkpoint_every", "dqn_epsilon_decay_steps",
                "eval_episodes", "ppo_rollout_steps"):
        assert protocol[key] == curves.PROTOCOL[key]
    for key in ("settings", "held_out", "training_setting", "eval_seed_base"):
        assert protocol[key] == difficulty.PROTOCOL[key]
    assert protocol["curve_eval_seed_base"] == 1000
    assert curves.PROTOCOL["threshold"] == 5.0
    assert protocol["sb3_dqn_exploration_fraction"] * protocol["total_steps"] == 3584


@pytest.mark.parametrize("change", [
    {"training_seeds": [0, 1, 2, 3]},
    {"training_seeds": [0, 1, 2, 3, 3]},
    {"sb3_dqn_exploration_fraction": 0.7},
    {"checkpoint_every": 768},
    {"held_out": ["no_bombs"]},
    {"eval_seed_base": 1000},
    {"agents": ["sb3_dqn"]},
])
def test_bad_designs_are_rejected(change):
    protocol = copy.deepcopy(sb3.PROTOCOL)
    protocol.update(change)
    with pytest.raises(ValueError):
        sb3.check_protocol(protocol)


def test_committed_protocol_file_matches_the_constant():
    saved = sb3.load_protocol()
    assert saved["name"] == "slicer-sb3-reference-v1"
    assert set(saved["frozen_inputs_sha256"]) == set(sb3.FROZEN_INPUTS)


def test_protocol_is_never_overwritten_and_drift_is_refused(tmp_path):
    for rel in sb3.FROZEN_INPUTS.values():
        (tmp_path / rel).parent.mkdir(parents=True, exist_ok=True)
        shutil.copy(sb3.ROOT / rel, tmp_path / rel)
    path = tmp_path / "protocol.json"
    sb3.write_protocol(path, root=tmp_path)
    with pytest.raises(FileExistsError):
        sb3.write_protocol(path, root=tmp_path)
    sb3.load_protocol(path, root=tmp_path)
    body = json.loads(path.read_text())
    body["total_steps"] = 40960
    path.write_text(json.dumps(body))
    with pytest.raises(RuntimeError):
        sb3.load_protocol(path, root=tmp_path)
    path.unlink()
    sb3.write_protocol(path, root=tmp_path)
    (tmp_path / sb3.FROZEN_INPUTS["curve_evidence"]).write_text("{}")
    with pytest.raises(RuntimeError):
        sb3.load_protocol(path, root=tmp_path)


def test_dqn_epsilon_reaches_its_floor_at_the_absolute_step():
    protocol = sb3.PROTOCOL
    model, env = sb3.build_model("sb3_dqn", 0, protocol, **TEST_SETTING)
    total, decay = protocol["total_steps"], protocol["dqn_epsilon_decay_steps"]
    schedule = model.exploration_schedule
    assert schedule(1.0) == 1.0
    assert schedule(1 - (decay - 1) / total) > 0.05
    assert schedule(1 - (decay + 1) / total) == 0.05
    assert abs(schedule(1 - decay / total) - 0.05) < 1e-12
    env.close()


@pytest.mark.parametrize("agent", ["sb3_dqn", "sb3_ppo"])
def test_callback_sees_one_evolving_model_and_does_not_change_training(agent):
    protocol = tiny_protocol()
    seen = []

    def record(step, model):
        seen.append((step, id(model), curves.fingerprint(model.policy)))
        eval_sb3(model, seed=4242, episodes=2, **TEST_SETTING)

    with_eval = sb3.train_sb3(agent, 3, protocol, record=record, **TEST_SETTING)
    without = sb3.train_sb3(agent, 3, protocol, **TEST_SETTING)
    assert [s for s, _, _ in seen] == [0, 512, 1024]
    assert {i for _, i, _ in seen} == {id(with_eval)}
    assert len({f for _, _, f in seen}) == 3
    assert curves.fingerprint(with_eval.policy) == curves.fingerprint(without.policy) == seen[-1][2]


def test_tiny_end_to_end_run_on_a_test_only_setting(tmp_path):
    protocol = tiny_protocol()
    path = tmp_path / "protocol.json"
    for rel in sb3.FROZEN_INPUTS.values():
        (tmp_path / rel).parent.mkdir(parents=True, exist_ok=True)
        shutil.copy(sb3.ROOT / rel, tmp_path / rel)
    sb3.write_protocol(path, protocol, root=tmp_path)
    result = sb3.run(path, tmp_path / "out.json", protocol_constant=protocol, root=tmp_path,
                     compare_with=None, full_checks=False)
    assert len(result["rows"]) == 20
    summary = result["curve"]["summary"]
    assert set(summary) == {"sb3_dqn", "sb3_ppo"} and summary["sb3_dqn"]["seeds"] == 5
    assert [c["step"] for c in summary["sb3_ppo"]["curve"]] == [0, 512, 1024]
    entry = result["difficulty_summary"]["agents"]["sb3_dqn"]["settings"]["test_only"]
    assert entry["transfer_gap"]["reading"] in ("transfer loss", "transfer gain",
                                                "no measurable difference")
    assert json.loads((tmp_path / "out.json").read_text())["protocol"] == "test-only"


def test_implementation_gap_reading_rule():
    assert sb3.implementation_gap([5, 5.1, 5.2, 5.3, 5.4], [1, 1.1, 1.2, 1.3, 1.4])["reading"] == "SB3 higher"
    assert sb3.implementation_gap([1, 1.1, 1.2, 1.3, 1.4], [5, 5.1, 5.2, 5.3, 5.4])["reading"] == "scratch higher"
    assert sb3.implementation_gap([1, 5, 1, 5, 3], [3, 1, 5, 1, 5])["reading"] == "no measurable difference"
