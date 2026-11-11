"""The new views and layout sets: base game untouched, layout visible, sets checked for leakage."""

import random

import numpy as np
import pytest
import torch

from agents import layout_ppo
from envs.gridquest import GridQuest, Layout, default_layout
from envs.gridquest_views import (ABSOLUTE_SIZE, BLOCKED, COIN, DOOR, EGOCENTRIC_SHAPE, GOAL, HAS_KEY,
                                  HAZARD, KEY, RADIUS, LayoutPool, ViewedGridQuest, path_score,
                                  shortest_solution)
from playtest.bug_hunt import _scripted_action
from playtest.levels import generate
from playtest.transfer_layouts import (LayoutRegistry, build_sets, content_hash, disjointness_report,
                                       draw_set, labelled_grid, reserved_layouts, set_hash, symmetries)

TEST_ONLY_SEED = 31337


def rejections() -> dict:
    return {"exact": 0, "near": 0, "invalid": 0, "unsolvable": 0}


def drawn(count: int, seed: int = TEST_ONLY_SEED) -> list[Layout]:
    return draw_set(random.Random(seed), count, LayoutRegistry(), rejections())


def test_integer_observation_is_unchanged_and_blind_to_the_layout() -> None:
    first, second = default_layout(), generate(20261020, 1)[0]
    env_a, env_b = GridQuest(layout=first), GridQuest(layout=second)
    env_a.position = env_b.position = (3, 3)
    assert env_a.observation_space.n == env_b.observation_space.n == 256
    assert env_a._obs() == env_b._obs()
    assert GridQuest().reset()[0] == 36


def test_egocentric_view_centres_the_player_and_shows_the_layout() -> None:
    layout = default_layout()
    game = ViewedGridQuest(layout)
    obs = game.reset()
    assert obs.shape == EGOCENTRIC_SHAPE and obs.dtype == np.float32
    def cell(position):
        x, y = position
        return RADIUS + y - game.env.position[1], RADIUS + x - game.env.position[0]
    assert obs[(KEY, *cell(layout.key_at))] == 1
    assert obs[(GOAL, *cell(layout.goal_at))] == 1
    assert obs[(DOOR, *cell(layout.door_at))] == 1
    assert obs[(COIN, *cell(layout.coin_at))] == 1
    for wall in layout.walls:
        assert obs[(BLOCKED, *cell(wall))] == 1
    for hazard in layout.hazards:
        assert obs[(HAZARD, *cell(hazard))] == 1
    assert obs[BLOCKED, RADIUS, RADIUS - 1] == 1 and obs[BLOCKED, RADIUS - 1, RADIUS] == 1
    assert obs[BLOCKED, RADIUS, RADIUS] == 0


def test_view_shifts_with_movement_and_tracks_key_and_coin() -> None:
    layout = default_layout()
    game = ViewedGridQuest(layout)
    before = game.reset()
    after, *_ = game.step(1)  # Down one cell.
    assert np.array_equal(after[GOAL, :-1, :], before[GOAL, 1:, :])
    game.env.position = (layout.key_at[0], layout.key_at[1] - 1)
    after, *_ = game.step(1)
    assert game.env.key and after[KEY].sum() == 0 and after[HAS_KEY].all()
    game.env.position = (layout.coin_at[0], layout.coin_at[1] + 1)
    after, *_ = game.step(0)
    assert game.env.coin and after[COIN].sum() == 0


def test_same_local_scene_gives_the_same_view_on_different_absolute_cells() -> None:
    base = {"size": 8, "walls": frozenset(), "hazards": frozenset(), "wall_gap": (-1, 0),
            "softlock_at": (7, 0)}
    one = Layout(start=(2, 2), key_at=(3, 2), coin_at=(2, 3), door_at=(4, 3), goal_at=(4, 2), **base)
    two = Layout(start=(3, 3), key_at=(4, 3), coin_at=(3, 4), door_at=(5, 4), goal_at=(5, 3), **base)
    first, second = ViewedGridQuest(one).reset(), ViewedGridQuest(two).reset()
    for plane in (KEY, COIN, DOOR, GOAL):
        assert np.array_equal(first[plane], second[plane])
    absolute_one = ViewedGridQuest(one, "absolute").reset()
    absolute_two = ViewedGridQuest(two, "absolute").reset()
    assert absolute_one.shape == (ABSOLUTE_SIZE,) and not np.array_equal(absolute_one, absolute_two)


def test_absolute_view_cannot_tell_two_layouts_apart() -> None:
    first, second = drawn(6)[0], drawn(6)[3]  # Both size 6.
    game_one, game_two = ViewedGridQuest(first, "absolute"), ViewedGridQuest(second, "absolute")
    game_one.reset(), game_two.reset()
    game_one.env.position = game_two.env.position = (1, 1)
    assert np.array_equal(game_one.observe(), game_two.observe())
    assert not np.array_equal(ViewedGridQuest(first).observe(), ViewedGridQuest(second).observe())


def test_viewed_game_matches_the_base_game_move_for_move() -> None:
    layout = generate(20261020, 1)[0]
    base, viewed = GridQuest(layout=layout, reward_mode="shaped"), ViewedGridQuest(layout, reward_mode="shaped")
    base.reset(), viewed.reset()
    rng = random.Random(4)
    for _ in range(200):
        action = rng.randrange(4)
        _, reward, terminated, truncated, info = base.step(action)
        _, viewed_reward, viewed_terminated, viewed_truncated, viewed_info = viewed.step(action)
        assert (reward, terminated, truncated, info) == (viewed_reward, viewed_terminated,
                                                         viewed_truncated, viewed_info)
        if terminated or truncated:
            break


def test_pool_draw_is_seeded_and_observation_fits_its_space() -> None:
    layouts = drawn(6)
    def order(seed):
        env = LayoutPool(layouts)
        env.reset(seed=seed)
        return [env.reset()[1]["layout"] for _ in range(20)]
    assert order(1) == order(1) and order(1) != order(2)
    pool = LayoutPool(layouts)
    assert pool.observation_space.contains(pool.reset(seed=0)[0])
    with pytest.raises(ValueError):
        LayoutPool([])


@pytest.mark.parametrize("layout", [default_layout(), *generate(20261020, 8)])
def test_scripted_path_equals_the_shortest_solution(layout: Layout) -> None:
    env = GridQuest(layout=layout)
    env.reset()
    for steps in range(1, env.max_steps + 1):
        *_, terminated, _, info = env.step(_scripted_action(env))
        if terminated:
            break
    assert info["success"] and steps == shortest_solution(layout)


def test_shortest_solution_obeys_the_door_and_reports_unsolvable_maps() -> None:
    layout = default_layout()
    assert shortest_solution(layout) == 10
    walled = Layout(**{**layout.__dict__, "walls": frozenset({(0, 1), (1, 0)})})
    assert shortest_solution(walled) is None


def test_path_score_rewards_only_short_successes() -> None:
    assert path_score(True, 10, 10) == 1.0
    assert path_score(True, 10, 20) == 0.5
    assert path_score(False, 10, 10) == 0.0


def test_generation_is_seeded_and_every_map_is_solvable() -> None:
    assert set_hash(drawn(30)) == set_hash(drawn(30))
    assert set_hash(drawn(30)) != set_hash(drawn(30, TEST_ONLY_SEED + 1))
    assert all(shortest_solution(layout) for layout in drawn(30))
    assert [layout.size for layout in drawn(6)] == [6, 7, 8, 6, 7, 8]


def test_content_hash_ignores_defect_only_cells() -> None:
    layout = drawn(1)[0]
    moved = Layout(**{**layout.__dict__, "softlock_at": (0, 0), "wall_gap": (-1, 0)})
    other = Layout(**{**layout.__dict__, "coin_at": (layout.size - 1, layout.size - 1)})
    assert content_hash(layout) == content_hash(moved)
    assert other == layout or content_hash(layout) != content_hash(other)


def test_near_duplicates_are_rejected_under_symmetry() -> None:
    layout = drawn(1)[0]
    registry = LayoutRegistry()
    registry.add(layout)
    assert registry.conflict(layout) == "exact"
    size = layout.size - 1
    def turn(cell):
        return (size - cell[0], cell[1])
    mirrored = Layout(size=layout.size, start=turn(layout.start), key_at=turn(layout.key_at),
                      coin_at=turn(layout.coin_at), door_at=turn(layout.door_at),
                      goal_at=turn(layout.goal_at), walls=frozenset(map(turn, layout.walls)),
                      hazards=frozenset(map(turn, layout.hazards)), wall_gap=layout.wall_gap,
                      softlock_at=layout.softlock_at)
    assert any(np.array_equal(view, labelled_grid(mirrored)) for view in symmetries(labelled_grid(layout)))
    assert registry.distance(mirrored) == 0 and registry.conflict(mirrored) == "near"


def test_build_sets_rejects_a_set_overlap_and_the_report_would_catch_one() -> None:
    plan = {"sets": {"train": {"seed": 1, "count": 12}, "test": {"seed": 2, "count": 12}},
            "smoke_seed": 3, "smoke_count": 5}
    built = build_sets(plan)
    report = disjointness_report(built["sets"], built["reserved"])
    assert report["all_disjoint"] and report["pairs"]["train|test"]["shared_hashes"] == 0
    assert report["pairs"]["train|test"]["min_cell_distance"] > report["threshold_cells"]
    assert set(built["reserved"]) == {"default", "earlier_eight", "smoke"}
    leaked = {"train": built["sets"]["train"], "test": [*built["sets"]["test"], built["sets"]["train"][0]]}
    report = disjointness_report(leaked, built["reserved"])
    assert not report["all_disjoint"] and report["pairs"]["train|test"]["shared_hashes"] == 1
    # The same seed twice cannot leak either: the registry rejects every repeat.
    again = build_sets({"sets": {"train": {"seed": 1, "count": 6}, "test": {"seed": 1, "count": 6}}})
    assert not {content_hash(layout) for layout in again["sets"]["train"]} & {
        content_hash(layout) for layout in again["sets"]["test"]}
    assert again["rejected"]["test"]["exact"] + again["rejected"]["test"]["near"] > 0


def test_reserved_layouts_hold_the_earlier_eight() -> None:
    assert reserved_layouts()["earlier_eight"] == generate(20261020, 8)


@pytest.mark.parametrize("view", ["egocentric", "absolute"])
def test_training_is_seeded_and_changes_the_model(view: str) -> None:
    torch.set_num_threads(1)
    layouts = drawn(3)
    first, info = layout_ppo.train(layouts, 5, 4096, view=view)
    second, _ = layout_ppo.train(layouts, 5, 4096, view=view)
    other, _ = layout_ppo.train(layouts, 6, 4096, view=view)
    untrained = layout_ppo.build_model(view)
    assert info["steps"] == 4096
    assert layout_ppo.fingerprint(first) == layout_ppo.fingerprint(second)
    assert layout_ppo.fingerprint(first) != layout_ppo.fingerprint(other)
    assert layout_ppo.fingerprint(first) != layout_ppo.fingerprint(untrained)
    assert sum(item["episodes"] for item in info["history"]) >= 0


def test_evaluation_is_per_layout_seeded_and_scores_a_perfect_player() -> None:
    layouts = drawn(4)
    one = layout_ppo.evaluate(None, layouts, 3, 2)
    assert one == layout_ppo.evaluate(None, layouts, 3, 2)
    assert len(one["spl"]) == len(one["success"]) == 4
    assert all(0 <= spl <= success <= 1 for spl, success in zip(one["spl"], one["success"]))
    assert one["shortest"] == [shortest_solution(layout) for layout in layouts]
    model = layout_ppo.build_model("egocentric")
    sampled = layout_ppo.evaluate(model, layouts, 3, 1)
    greedy = layout_ppo.evaluate(model, layouts, 3, 1, mode="greedy")
    assert len(sampled["spl"]) == len(greedy["spl"]) == 4
    with pytest.raises(ValueError):
        layout_ppo.evaluate(None, layouts, 3, 1, mode="greedy")
