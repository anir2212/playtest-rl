"""Slicer difficulty protocol: structure, guards, anchors and a tiny end-to-end run.

No test trains or evaluates an agent on a held-out setting; the small run uses a
test-only setting that is not part of the protocol.
"""

import copy
import json
import shutil

import pytest

from analysis import curves, difficulty
from analysis.figures import difficulty_figure
from envs.slicer import Slicer


def tiny_protocol():
    protocol = copy.deepcopy(difficulty.PROTOCOL)
    protocol.update({"name": "test-only", "total_steps": 512, "checkpoint_every": 512,
                     "eval_episodes": 2,
                     "settings": {"train": {"bomb_ratio": 0.2, "speed": 1.0},
                                  "test_only": {"bomb_ratio": 0.1, "speed": 1.0}},
                     "held_out": ["test_only"]})
    return protocol


def test_protocol_is_well_formed_and_trains_on_the_default_game():
    protocol = difficulty.PROTOCOL
    difficulty.check_protocol(protocol)
    default = Slicer()
    assert protocol["settings"]["train"] == {"bomb_ratio": default.bomb_ratio, "speed": default.speed}
    assert len(protocol["held_out"]) == 7 and "train" not in protocol["held_out"]
    for key in ("training_seeds", "total_steps", "checkpoint_every", "dqn_epsilon_decay_steps",
                "eval_episodes"):
        assert protocol[key] == curves.PROTOCOL[key]
    assert protocol["consistency_eval_seed_base"] == 1000 != protocol["eval_seed_base"]


def test_committed_protocol_file_matches_the_constant():
    if not difficulty.PROTOCOL_PATH.exists():
        pytest.skip("protocol not written yet")
    assert difficulty.load_protocol()["name"] == difficulty.PROTOCOL["name"]


@pytest.mark.parametrize("change", [
    {"training_seeds": [0, 1, 2, 3]},
    {"training_seeds": [0, 1, 2, 3, 3]},
    {"held_out": ["train", "test_only"]},
    {"held_out": []},
    {"settings": {"train": {"bomb_ratio": 0.2, "speed": 1.0},
                  "test_only": {"bomb_ratio": 0.2, "speed": 1.0}}},
    {"eval_seed_base": 1000},
])
def test_check_protocol_rejects_bad_designs(change):
    protocol = {**tiny_protocol(), **change}
    with pytest.raises(ValueError):
        difficulty.check_protocol(protocol)


def test_protocol_is_never_overwritten_and_drift_is_refused(tmp_path):
    path, curve = tmp_path / "protocol.json", tmp_path / "curve.json"
    shutil.copy(difficulty.CURVE_EVIDENCE, curve)
    protocol = tiny_protocol()
    difficulty.write_protocol(path, protocol, curve)
    with pytest.raises(FileExistsError):
        difficulty.write_protocol(path, protocol, curve)
    assert difficulty.load_protocol(path, protocol, curve)["name"] == "test-only"
    changed = copy.deepcopy(protocol)
    changed["settings"]["test_only"]["speed"] = 1.1
    with pytest.raises(RuntimeError):
        difficulty.load_protocol(path, changed, curve)
    curve.write_text(curve.read_text() + " ")
    with pytest.raises(RuntimeError):
        difficulty.load_protocol(path, protocol, curve)


def test_scripted_anchor_beats_random_on_the_training_setting():
    scripted = difficulty.evaluate_anchor("scripted", seed=3, episodes=3, bomb_ratio=0.2, speed=1.0)
    random = difficulty.evaluate_anchor("random", seed=3, episodes=3, bomb_ratio=0.2, speed=1.0)
    assert scripted["mean_score"] > random["mean_score"] + 3
    assert scripted["mean_bomb_hits"] == 0


def test_gap_reading_follows_the_interval():
    assert difficulty.gap_ci([5, 6, 5, 6, 5], [1, 2, 1, 2, 1])["reading"] == "transfer loss"
    assert difficulty.gap_ci([1, 2, 1, 2, 1], [5, 6, 5, 6, 5])["reading"] == "transfer gain"
    assert difficulty.gap_ci([1, 5, 1, 5, 3], [2, 4, 3, 1, 5])["reading"] == "no measurable difference"


def test_tiny_run_records_seeds_gaps_and_rejects_duplicate_seeds(tmp_path):
    protocol = tiny_protocol()
    path = tmp_path / "protocol.json"
    difficulty.write_protocol(path, protocol)
    result = difficulty.run(path, tmp_path / "out.json", protocol_constant=protocol,
                            compare_with=None)
    rows = result["rows"]
    assert len(rows) == 2 * 5 * 2
    assert result["curve_consistency"]["identical"] is None
    for row in rows:
        assert row["eval_seed"] == 7000 + row["seed"]
        assert [c["step"] for c in row["checkpoints"]] == [0, 512]
    primary = [r for r in rows if r["train_setting"] == "train"]
    assert all(set(r["checkpoints"][0]["eval"]) == {"train", "test_only"} for r in primary)
    assert all(r["consistency_eval_seed"] == 1000 + r["seed"] for r in primary)
    entry = result["summary"]["agents"]["double_dqn"]["settings"]["test_only"]
    assert len(entry["mean_score"]["seed_values"]) == 5 and "transfer_gap" in entry
    assert json.loads((tmp_path / "out.json").read_text())["protocol"] == "test-only"
    duplicated = [dict(r, seed=0) if r["seed"] == 1 else r for r in rows]
    with pytest.raises(ValueError):
        difficulty.final_values(duplicated, "double_dqn", "train", "train")
    evidence = tmp_path / "evidence.json"
    evidence.write_text(json.dumps({**result, "protocol": "slicer-difficulty-v1",
                                    "curve_consistency": {"identical": True}}))
    difficulty_figure(evidence, tmp_path / "figure.png")
    assert (tmp_path / "figure.png").stat().st_size > 0
    evidence.write_text(json.dumps(result))
    with pytest.raises(ValueError):
        difficulty_figure(evidence, tmp_path / "other.png")
