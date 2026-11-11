"""Guarded v3 runner: hash guards, hunt-loop agreement with the frozen hunt, acceptance logic."""

import json
import shutil
from pathlib import Path

import pytest

from envs.gridquest import default_layout
from playtest.bug_hunt import hunt
from playtest.reward_oracle import hunt_v3
from playtest.reward_oracle_run import (PROTOCOL_REL, _key, acceptance, controls, load_protocol,
                                        render)

ROOT = Path(__file__).resolve().parents[1]
PROTOCOL = json.loads((ROOT / PROTOCOL_REL).read_text())
POLICIES = PROTOCOL["policies"]


def copy_protocol_inputs(tmp_path: Path) -> None:
    for rel in [PROTOCOL_REL, *PROTOCOL["frozen_inputs_sha256"]]:
        destination = tmp_path / rel
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(ROOT / rel, destination)


def test_guard_rejects_a_changed_protocol_or_frozen_input(tmp_path: Path) -> None:
    copy_protocol_inputs(tmp_path)
    assert load_protocol(tmp_path)["name"] == "gridquest-reward-oracle-v1"
    for rel in ("playtest/oracle.py", "playtest/reward_oracle.py", "envs/gridquest.py"):
        original = (tmp_path / rel).read_text()
        (tmp_path / rel).write_text(original + "\n# changed\n")
        with pytest.raises(RuntimeError, match=rel):
            load_protocol(tmp_path)
        (tmp_path / rel).write_text(original)
    (tmp_path / PROTOCOL_REL).write_text((ROOT / PROTOCOL_REL).read_text() + "\n")
    with pytest.raises(RuntimeError, match="committed hash"):
        load_protocol(tmp_path)


@pytest.mark.parametrize("policy", POLICIES)
def test_clean_hunt_loop_matches_the_frozen_hunt(policy: str) -> None:
    v2 = hunt(None, policy, 7, 600, oracle="v2")
    v3 = hunt_v3(None, policy, 7, 600)
    assert v3["detected"] is None and v2["detected"] is None
    assert v3["position_visits"] == v2["position_visits"]
    assert v3["position_coverage"] == v2["position_coverage"]


def test_reward_fault_is_reported_at_first_success_and_v2_stays_silent() -> None:
    v3 = hunt_v3("goal_reward_missing", "scripted", 0, 200, layout=default_layout())
    assert v3["detected"] == "reward_mismatch"
    assert v3["time_to_discovery"] == v3["first_success_step"] == 10
    assert hunt("goal_reward_missing", "scripted", 0, 200, oracle="v2")["detected"] is None


def row(oracle: str, defect: str | None, policy: str = "scripted", seed: int = 0, map_index: int = 0,
        step: int | None = None, kind: str | None = None, success: int | None = None) -> dict:
    return {"map_set": "default", "map": map_index, "defect": defect, "oracle": oracle,
            "policy": policy, "seed": seed, "time_to_discovery": step, "detected": kind,
            "position_coverage": 0.5, "first_success_step": success}


def passing_rows() -> list[dict]:
    rows = []
    for seed in range(45):
        for policy in POLICIES:
            rows.append(row("v3", None, policy, seed))
            rows.append(row("v2", "goal_reward_missing", policy, seed))
            exposed = policy == "scripted"
            rows.append(row("v3", "goal_reward_missing", policy, seed,
                            step=10 if exposed else None, kind="reward_mismatch" if exposed else None,
                            success=10 if exposed else None))
            for defect in PROTOCOL["defects"]["earlier_catalogue"]:
                rows.append(row("v2", defect, policy, seed, step=5, kind="position_mismatch"))
                rows.append(row("v3", defect, policy, seed, step=5, kind="position_mismatch"))
    return rows


def verdict(rows: list[dict], control_ok: bool = True, loop_ok: bool = True) -> dict:
    return acceptance(rows, PROTOCOL, {"check": {"all_identical": control_ok}},
                      {"all_identical": loop_ok})


def test_acceptance_passes_only_when_every_rule_holds() -> None:
    assert verdict(passing_rows())["overall"] == "PASS"
    assert verdict(passing_rows(), control_ok=False)["D_controls"] is False
    assert verdict(passing_rows(), loop_ok=False)["overall"] == "FAIL"


@pytest.mark.parametrize("change,rule", [
    (lambda r: r["oracle"] == "v3" and r["defect"] is None and r["seed"] == 3, "A_no_false_alarms"),
    (lambda r: r["oracle"] == "v2" and r["defect"] == "goal_reward_missing" and r["seed"] == 3,
     "D_controls"),
])
def test_any_clean_or_control_alarm_fails_its_rule(change, rule) -> None:
    rows = passing_rows()
    hit = next(r for r in rows if change(r))
    hit.update(time_to_discovery=4, detected="reward_mismatch")
    outcome = verdict(rows)
    assert outcome[rule] is False and outcome["overall"] == "FAIL"


def test_exposed_silence_late_report_and_lost_detection_fail() -> None:
    rows = passing_rows()
    hit = next(r for r in rows if r["oracle"] == "v3" and r["policy"] == "random"
               and r["defect"] == "goal_reward_missing")
    hit["first_success_step"] = 50
    assert verdict(rows)["B_reward_fault"] is False
    hit.update(time_to_discovery=60, detected="reward_mismatch")
    assert verdict(rows)["B_reward_fault"] is False

    rows = passing_rows()
    hit = next(r for r in rows if r["oracle"] == "v3" and r["defect"] == "softlock")
    hit.update(time_to_discovery=None, detected=None)
    assert verdict(rows)["C_no_regression"] is False


def test_censored_hunter_that_never_reached_the_fault_does_not_fail_the_oracle() -> None:
    assert verdict(passing_rows())["B_reward_fault"] is True  # Three hunters here never reach it.


def test_controls_flag_any_row_that_differs_from_committed_evidence() -> None:
    rows = [row("v2", None, seed=seed) for seed in range(3)]
    reference = [dict(r) for r in rows]
    groups = {"src": {"clean": reference}}
    assert controls(rows, groups)["src|clean"]["all_identical"] is True
    reference[1]["time_to_discovery"] = 9
    assert controls(rows, groups)["src|clean"]["all_identical"] is False
    assert controls(rows[:2], {"src": {"clean": rows}})["src|clean"]["all_identical"] is False


def test_rendered_results_are_valid_json_with_one_line_per_row() -> None:
    result = {"protocol": "x", "rows": [row("v3", None), row("v2", None)]}
    text = render(result)
    assert json.loads(text) == result
    assert _key(result["rows"][0]) == ("default", 0, "scripted", 0)
    assert len([line for line in text.splitlines() if '"map_set"' in line]) == 2
