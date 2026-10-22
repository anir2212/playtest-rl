import json

import pytest

from agents import dqn, ppo
from analysis import curves
from analysis.figures import curve_figure


def test_dqn_checkpoints_share_one_evolving_model():
    seen = []
    model, log = dqn.train(0, steps=256, checkpoint_every=128,
                           on_checkpoint=lambda step, m: seen.append((step, id(m), curves.fingerprint(m))))
    assert [s for s, _, _ in seen] == [0, 128, 256]
    assert {i for _, i, _ in seen} == {id(model)}
    assert seen[0][2] != seen[-1][2]
    assert seen[-1][2] == curves.fingerprint(model)
    assert log["epsilon_decay_steps"] == int(256 * 0.7)


def test_ppo_checkpoints_follow_rollouts():
    seen = []
    model, _ = ppo.train(0, steps=1024, checkpoint_every=512,
                         on_checkpoint=lambda step, m: seen.append((step, id(m))))
    assert seen == [(0, id(model)), (512, id(model)), (1024, id(model))]


@pytest.mark.parametrize("agent", ["double_dqn", "scratch_ppo"])
def test_in_place_evaluation_does_not_change_training(agent):
    train, evaluate = curves.AGENTS[agent]
    steps, every = (256, 128) if agent == "double_dqn" else (1024, 512)
    plain, _ = train(3, steps=steps)
    checked, _ = train(3, steps=steps, checkpoint_every=every,
                       on_checkpoint=lambda _s, m: evaluate(m, seed=1003, episodes=2))
    assert curves.fingerprint(plain) == curves.fingerprint(checked)


def test_absolute_epsilon_default_matches_historical_schedule():
    legacy, _ = dqn.train(1, steps=300)
    explicit, _ = dqn.train(1, steps=300, epsilon_decay_steps=int(300 * 0.7))
    different, _ = dqn.train(1, steps=300, epsilon_decay_steps=20)
    assert curves.fingerprint(legacy) == curves.fingerprint(explicit)
    assert curves.fingerprint(legacy) != curves.fingerprint(different)


def test_checkpoint_argument_validation():
    with pytest.raises(ValueError):
        dqn.train(0, steps=256, checkpoint_every=100, on_checkpoint=lambda *_: None)
    with pytest.raises(ValueError):
        dqn.train(0, steps=256, checkpoint_every=128)
    with pytest.raises(ValueError):
        dqn.train(0, steps=256, epsilon_decay_steps=0)
    with pytest.raises(ValueError):
        ppo.train(0, steps=1024, checkpoint_every=256, on_checkpoint=lambda *_: None)
    with pytest.raises(ValueError):
        ppo.train(0, steps=1536, checkpoint_every=1024, on_checkpoint=lambda *_: None)


def _run(agent, seed, scores):
    return {"agent": agent, "seed": seed, "checkpoints": [
        {"step": step, "mean_score": score, "mean_bomb_hits": 0.0}
        for step, score in zip((0, 10, 20), scores)]}


def test_threshold_is_first_crossing_and_misses_are_censored():
    rows = [_run("a", 0, [0, 6, 4]), _run("a", 1, [0, 1, 2])]
    summary = curves.summarize(rows, threshold=5.0)["a"]
    assert summary["steps_to_threshold_by_seed"] == {0: 10, 1: None}
    assert summary["reached_threshold"] == 1 and summary["censored_at_steps"] == 20
    assert summary["curve"][1]["mean_score"]["mean"] == 3.5


def test_summary_rejects_mismatched_steps_and_duplicate_seeds():
    bad = _run("a", 1, [0, 1, 2])
    bad["checkpoints"][1]["step"] = 11
    with pytest.raises(ValueError):
        curves.summarize([_run("a", 0, [0, 1, 2]), bad], 5.0)
    with pytest.raises(ValueError):
        curves.summarize([_run("a", 0, [0, 1, 2]), _run("a", 0, [0, 1, 2])], 5.0)


def test_small_run_records_protocol_mismatch_and_eval_seeds():
    result = curves.run_curve(seeds=1, steps=512, every=512, eval_episodes=2)
    assert result["matches_protocol"] is False
    assert {r["agent"] for r in result["rows"]} == {"double_dqn", "scratch_ppo"}
    for row in result["rows"]:
        assert row["eval_seed"] == 1000 and [c["step"] for c in row["checkpoints"]] == [0, 512]


def test_endpoint_consistency_compares_seed_by_seed():
    curve = {"rows": [_run("a", 0, [0, 1, 2]), _run("a", 1, [0, 3, 4])]}
    endpoint = {"rows": [{"agent": "a", "seed": 0, "eval": {"mean_score": 1}},
                         {"agent": "a", "seed": 1, "eval": {"mean_score": 3.5}}]}
    check = curves.endpoint_consistency(curve, endpoint, "a", 10)
    assert check["identical"] is False and len(check["pairs"]) == 2
    with pytest.raises(ValueError):
        curves.endpoint_consistency(curve, endpoint, "a", 15)


def test_curve_figure_requires_protocol_evidence(tmp_path):
    summary = curves.summarize([_run("double_dqn", 0, [0, 1, 6]), _run("double_dqn", 1, [0, 2, 3])], 5.0)
    data = {"protocol": curves.PROTOCOL, "matches_protocol": True, "summary": summary}
    path = tmp_path / "curve.json"
    path.write_text(json.dumps(data))
    curve_figure(path, tmp_path / "curve.png")
    assert (tmp_path / "curve.png").stat().st_size > 0
    path.write_text(json.dumps({**data, "matches_protocol": False}))
    with pytest.raises(ValueError):
        curve_figure(path, tmp_path / "other.png")
