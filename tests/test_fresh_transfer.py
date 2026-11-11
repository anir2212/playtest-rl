"""Protocol guards and direction of the fresh transfer comparison; no model training."""

import copy

import pytest

from analysis import fresh_transfer


def test_frozen_protocol_builds_exactly_the_declared_jobs():
    protocol = fresh_transfer.load_protocol()
    jobs = fresh_transfer.build_jobs(protocol)
    keys = [(agent, job["train_setting"], job["seed"]) for agent, job in jobs]
    assert len(keys) == len(set(keys)) == 84
    assert set(protocol["held_out"]).isdisjoint({"no_bombs", "bombs_0.4", "bombs_0.6"})
    assert all(job["eval_settings"] == protocol["held_out"] for _, job in jobs
               if job["train_setting"] == "train")
    assert all(job["eval_settings"] == [job["train_setting"]] for _, job in jobs
               if job["train_setting"] != "train")


def test_primary_gap_uses_in_setting_minus_default_return():
    protocol = fresh_transfer.load_protocol()
    rows = []
    for agent in protocol["agents"]:
        for trained_on in protocol["settings"]:
            for seed in protocol["training_seeds"]:
                evaluations = {}
                for setting in protocol["held_out"] if trained_on == "train" else [trained_on]:
                    value = 2.0 if trained_on == "train" else 3.0
                    evaluations[setting] = {"mean_return": value, "mean_score": value,
                                            "mean_bomb_hits": 0.1}
                rows.append({"agent": agent, "train_setting": trained_on, "seed": seed,
                             "checkpoints": [{"step": step, "eval": evaluations}
                                             for step in (0, 20480, 40960)]})
    anchors = [{"setting": setting, "policy": policy, "seed": seed,
                "mean_score": 1.0, "mean_return": 1.0, "mean_bomb_hits": 0.0}
               for setting in protocol["held_out"] for policy in ("random", "scripted")
               for seed in protocol["training_seeds"]]
    summary = fresh_transfer.summarize(rows, anchors, protocol)
    assert summary["primary"]["gap"]["mean"] == 1.0
    assert summary["primary"]["gap"]["reading"] == "transfer loss"
    assert sum(map(len, summary["cells"].values())) == 8
    broken = copy.deepcopy(rows)
    broken.pop()
    with pytest.raises(ValueError, match="missing or repeated seed"):
        fresh_transfer.summarize(broken, anchors, protocol)
