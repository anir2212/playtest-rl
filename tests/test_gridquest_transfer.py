"""Guarded layout-transfer runner: hash guards, phase separation, intervals and the acceptance rule."""

import json
import shutil
from pathlib import Path

import numpy as np
import pytest

from analysis import gridquest_transfer as gt
from envs.gridquest import Layout
from playtest.transfer_layouts import content_hash, reserved_layouts, set_hash

ROOT = Path(__file__).resolve().parents[1]
PROTOCOL = json.loads((ROOT / gt.PROTOCOL_REL).read_text())


@pytest.fixture(scope="module")
def sets() -> dict[str, list[Layout]]:
    return gt.load_sets(gt.load_protocol())[0]


def copy_inputs(tmp_path: Path) -> None:
    for rel in [gt.PROTOCOL_REL, *PROTOCOL["frozen_inputs_sha256"]]:
        destination = tmp_path / rel
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(ROOT / rel, destination)


def test_guard_rejects_a_changed_protocol_or_frozen_input(tmp_path: Path) -> None:
    copy_inputs(tmp_path)
    assert gt.load_protocol(tmp_path)["name"] == "gridquest-layout-transfer-v1"
    for rel in ("envs/gridquest_views.py", "agents/layout_ppo.py", "playtest/transfer_layouts.py",
                "envs/gridquest.py"):
        original = (tmp_path / rel).read_text()
        (tmp_path / rel).write_text(original + "\n# changed\n")
        with pytest.raises(RuntimeError, match=rel):
            gt.load_protocol(tmp_path)
        (tmp_path / rel).write_text(original)
    (tmp_path / gt.PROTOCOL_REL).write_text((ROOT / gt.PROTOCOL_REL).read_text() + "\n")
    with pytest.raises(RuntimeError, match="committed hash"):
        gt.load_protocol(tmp_path)


def test_sets_regenerate_to_the_frozen_hashes_and_refuse_a_change(sets, monkeypatch) -> None:
    protocol = gt.load_protocol()
    for name, layouts in sets.items():
        assert set_hash(layouts) == protocol["layout_set_sha256"][name]
        assert len(layouts) == protocol["layout_plan"]["sets"][name]["count"]
    changed = json.loads(json.dumps(protocol))
    changed["layout_plan"]["sets"]["test"]["seed"] += 1
    with pytest.raises(RuntimeError, match="differs from the protocol: test"):
        gt.load_sets(changed)
    monkeypatch.setattr(gt, "disjointness_report", lambda *_: {"all_disjoint": False})
    with pytest.raises(RuntimeError, match="not disjoint"):
        gt.load_sets(protocol)


def test_frozen_sets_share_nothing_with_each_other_or_with_seen_maps(sets) -> None:
    protocol = gt.load_protocol()
    groups = {**sets, **reserved_layouts(protocol["layout_plan"]["smoke_seed"],
                                         protocol["layout_plan"]["smoke_count"])}
    hashes = {name: {content_hash(layout) for layout in layouts} for name, layouts in groups.items()}
    for name in sets:
        assert len(hashes[name]) == len(sets[name])
        assert all(not hashes[name] & other for key, other in hashes.items() if key != name)
    report = protocol["disjointness"]
    assert report["all_disjoint"] and report["threshold_cells"] == 4
    assert all(item["min_cell_distance"] > 4 for item in report["pairs"].values()
               if item["min_cell_distance"] is not None)


def test_search_jobs_never_receive_a_test_layout(sets) -> None:
    protocol = gt.load_protocol()
    jobs = gt.search_jobs(protocol, sets)
    test = {content_hash(layout) for layout in sets["test"]}
    assert len(jobs) == len(protocol["search_space"]) * len(protocol["search_seeds"])
    for job in jobs:
        assert set(job["eval_sets"]) == {"validation"}
        carried = job["layouts"] + job["eval_sets"]["validation"]
        assert not test & {content_hash(layout) for layout in carried}


def test_training_blocks_are_separate_per_seed_and_nested_per_count(sets) -> None:
    protocol = gt.load_protocol()
    width = max(protocol["training_layout_counts"])
    slots = len(protocol["search_seeds"]) + len(protocol["final_seeds"])
    blocks = [gt.training_block(protocol, sets, slot, width) for slot in range(slots)]
    assert len({content_hash(layout) for block in blocks for layout in block}) == slots * width
    assert gt.training_block(protocol, sets, 2, 10) == blocks[2][:10]
    with pytest.raises(ValueError):
        gt.training_block(protocol, sets, slots, width)


def test_final_jobs_use_only_the_chosen_setting(sets) -> None:
    protocol = gt.load_protocol()
    chosen = list(protocol["search_space"])[-1]
    jobs = gt.final_jobs(protocol, sets, chosen, reserved_layouts()["earlier_eight"])
    assert len(jobs) == len(protocol["conditions"]) * len(protocol["final_seeds"])
    assert {job["config"] for job in jobs} == {chosen}
    assert {job["train_seed"] for job in jobs} == set(protocol["final_seeds"])
    assert all(len(job["eval_sets"]["test"]) == len(sets["test"]) for job in jobs)
    assert all(len(job["layouts"]) == job["count"] for job in jobs)


def test_selection_uses_validation_mean_and_ties_go_to_the_earlier_setting() -> None:
    protocol = gt.load_protocol()
    mode, metric = protocol["selection"]["mode"], protocol["selection"]["metric"]
    names = list(protocol["search_space"])
    rows = []
    for position, config in enumerate(names):
        for seed in protocol["search_seeds"]:
            score = 0.5 if position in (1, 2) else 0.1
            rows.append({"config": config, "train_seed": seed,
                         "evaluations": {"validation": {mode: {metric: [score, score]}}}})
    assert gt.select(rows, protocol)["chosen"] == names[1]
    with pytest.raises(ValueError, match="incomplete"):
        gt.select(rows[1:], protocol)


def test_scripted_reference_solves_every_map_on_its_shortest_path(sets) -> None:
    outcome = gt.scripted_reference(sets["validation"][:12])
    assert outcome == {"layouts": 12, "solved": 12, "on_shortest_path": 12, "spl": 1.0}


def test_intervals_widen_with_seed_and_layout_noise_and_cover_the_mean() -> None:
    rng = np.random.default_rng(0)
    flat = np.full((5, 40), 0.5)
    assert gt.crossed_interval(flat, rng, 200) == [0.5, 0.5]
    noisy_seed = flat + rng.normal(0, 0.2, (5, 1))
    noisy_layout = flat + rng.normal(0, 0.2, (1, 40))
    for values in (noisy_seed, noisy_layout):
        low, high = gt.crossed_interval(values, rng, 500)
        assert low < values.mean() < high and high - low > 0.02


def test_contrast_pairs_layouts_and_gap_keeps_seeds() -> None:
    rng = np.random.default_rng(1)
    layout_effect = rng.normal(0.5, 0.2, (1, 60))
    first, second = layout_effect + 0.1, layout_effect
    assert gt.contrast_interval(np.tile(first, (5, 1)), np.tile(second, (5, 1)), rng, 300) == [0.1, 0.1]
    with pytest.raises(ValueError):
        gt.contrast_interval(np.zeros((5, 3)), np.zeros((5, 4)), rng, 10)
    seed_effect = rng.normal(0.5, 0.2, (5, 1))
    low, high = gt.gap_interval(seed_effect + 0.2, seed_effect, rng, 300)
    assert low == pytest.approx(0.2) and high == pytest.approx(0.2)
    with pytest.raises(ValueError):
        gt.gap_interval(np.zeros((4, 3)), np.zeros((5, 3)), rng, 10)


@pytest.mark.parametrize("margin,single,control,overall", [
    (0.2, 0.1, 0.1, "PASS"), (0.01, 0.1, 0.1, "FAIL"), (0.2, 0.0, 0.1, "FAIL"), (0.2, 0.1, -0.1, "FAIL")])
def test_acceptance_needs_all_three_lower_bounds(margin, single, control, overall) -> None:
    protocol = gt.load_protocol()
    required = protocol["acceptance_rule"]["agent_vs_random_lower_bound_at_least"]
    summary = {"contrasts": {
        "agent_vs_random": {"ci95": [required if margin == 0.2 else required - 0.01, 0.9]},
        "agent_vs_single_layout": {"ci95": [single, 0.9]},
        "agent_vs_absolute_control": {"ci95": [control, 0.9]}}}
    assert gt.acceptance(summary, protocol)["overall"] == overall


def test_tiny_job_end_to_end_gives_per_layout_rows(sets) -> None:
    protocol = gt.load_protocol()
    job = gt._job({**protocol, "steps": 2048}, "smoke", "egocentric_1", "egocentric", 1, 0, 9999,
                  list(protocol["search_space"])[0], sets, {"validation": sets["validation"][:3]})
    row = gt.train_job({**job, "eval_modes": {"sampled": 1, "greedy": 1}})["row"]
    assert row["consumed_steps"] == 2048 and row["count"] == 1
    assert len(row["evaluations"]["validation"]["greedy"]["spl"]) == 3
    assert len(row["param_sha256_16"]) == 16


def test_render_is_valid_json() -> None:
    result = {"protocol": "x", "rows": [{"a": 1}, {"a": 2}]}
    assert json.loads(gt.render(result)) == result


def synthetic_outcome(sets, protocol) -> tuple[list[dict], dict]:
    """Made-up scores in the real result shape, so the summary path runs without playing a map."""
    rng = np.random.default_rng(3)
    set_sizes = {"validation": 100, "test": 200, "earlier_eight": 8}
    def scores(size, level):
        return {"spl": np.clip(rng.normal(level, 0.05, size), 0, 1).round(4).tolist(),
                "success": np.clip(rng.normal(level + 0.1, 0.05, size), 0, 1).round(4).tolist()}
    rows = []
    for entry in protocol["conditions"]:
        level = 0.8 if entry["name"] == "egocentric_1000" else 0.1
        for seed in protocol["final_seeds"]:
            sizes = {**set_sizes, "own_training": entry["count"]}
            rows.append({"condition": entry["name"], "train_seed": seed, "evaluations": {
                name: {"sampled": scores(size, level), "greedy": scores(size, level)}
                for name, size in sizes.items()}})
    references = {"random": {name: {str(seed): scores(size, 0.05) for seed in protocol["final_seeds"]}
                             for name, size in set_sizes.items()}}
    return rows, references


def test_summary_and_verdict_run_on_the_real_shape_and_are_json_safe(sets) -> None:
    protocol = gt.load_protocol()
    protocol = {**protocol, "bootstrap": {**protocol["bootstrap"], "resamples": 50}}
    rows, references = synthetic_outcome(sets, protocol)
    summary = gt.summarize(rows, references, protocol, sets)
    verdict = gt.acceptance(summary, protocol)
    assert verdict["overall"] == "PASS"
    assert set(summary["contrasts"]) == set(protocol["primary_contrast"]["contrasts"])
    assert set(summary["generalisation_gap"]) == {entry["name"] for entry in protocol["conditions"]}
    cell = summary["cells"]["egocentric_1000"]["test"]["sampled_spl"]
    assert cell["seeds"] == len(protocol["final_seeds"]) and cell["layouts"] == 200
    assert summary["cells"]["egocentric_1"]["own_training"]["greedy_spl"]["layouts"] == 1
    assert "own_training" not in summary["cells"]["random"]
    assert summary["lead_agent_test_layouts"]["layouts"] == 200
    assert json.loads(json.dumps({"s": summary, "v": verdict}))["v"]["overall"] == "PASS"
