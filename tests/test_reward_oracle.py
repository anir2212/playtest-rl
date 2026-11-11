"""Contract v3 on hand-written transitions and clean play; no hunt or injected defect."""

import random

import pytest

from envs.gridquest import GridQuest, default_layout
from playtest.oracle import check_transition, public_state
from playtest.reward_oracle import REWARD_MODES, RewardRules, check_transition_v3, expected_reward

LAYOUT = default_layout()  # Key (0,5), coin (3,0), door (4,5), goal (5,5), hazard (4,1).


def state(position, key=False, coin=False, score=0):
    return {"position": position, "key": key, "coin": coin, "score": score}


def test_clean_table_for_every_reward_mode():
    for mode in REWARD_MODES:
        rules = RewardRules.from_layout(LAYOUT, mode)
        plain = expected_reward(rules, state((0, 0)), state((1, 0)))
        hazard = expected_reward(rules, state((3, 1)), state((4, 1)))
        goal = expected_reward(rules, state((4, 5), key=True), state((5, 5), key=True))
        assert plain == -0.01
        assert hazard == pytest.approx(-0.06)
        assert goal == 1.0


def test_key_bonus_follows_the_declared_rule_for_each_mode():
    first = (state((0, 4)), state((0, 5), key=True))
    reentry = (state((0, 4), key=True), state((0, 5), key=True))
    for mode, first_paid, reentry_paid in (("sparse", -0.01, -0.01), ("shaped", 0.19, -0.01),
                                           ("fixed", 0.19, -0.01), ("hackable", 0.19, 0.19)):
        rules = RewardRules.from_layout(LAYOUT, mode)
        assert expected_reward(rules, *first) == pytest.approx(first_paid)
        assert expected_reward(rules, *reentry) == pytest.approx(reentry_paid)


def test_goal_without_key_pays_only_the_step_cost():
    rules = RewardRules.from_layout(LAYOUT)
    assert expected_reward(rules, state((5, 4)), state((5, 5))) == -0.01


@pytest.mark.parametrize("before,action,after,wrong_reward", [
    (state((4, 5), key=True), 3, {**state((5, 5), key=True, score=10), "success": True}, 0.0),
    (state((4, 5), key=True), 3, {**state((5, 5), key=True, score=10), "success": True}, 2.0),
    (state((0, 0)), 1, state((0, 1)), 0.0),
    (state((4, 0), coin=True, score=1), 1, state((4, 1), coin=True, score=1), -0.01),
])
def test_reward_faults_with_correct_state_are_reported(before, action, after, wrong_reward):
    rules = RewardRules.from_layout(LAYOUT)
    assert check_transition(rules.state, before, action, after) is None
    assert check_transition_v3(rules, before, action, after, wrong_reward) == "reward_mismatch"


def test_state_faults_keep_their_v2_name_even_with_a_wrong_reward():
    rules = RewardRules.from_layout(LAYOUT)
    assert check_transition_v3(rules, state((0, 0)), 1, state((0, 0)), 5.0) == "position_mismatch"


def test_oracle_ignores_success_and_bug_labels():
    rules = RewardRules.from_layout(LAYOUT)
    after = {**state((5, 5), key=True, score=10), "success": False, "bug_detected": "wall_gap"}
    before = state((4, 5), key=True)
    assert check_transition_v3(rules, before, 3, after, 1.0) is None
    assert check_transition_v3(rules, before, 3, after, -0.01) == "reward_mismatch"


def test_unknown_reward_mode_is_rejected():
    with pytest.raises(ValueError):
        RewardRules.from_layout(LAYOUT, "made_up")


@pytest.mark.parametrize("mode", REWARD_MODES)
def test_clean_random_play_matches_the_table_in_every_mode(mode):
    """Seeds 1000+ on the default map only: development play, outside any hunt protocol."""
    hazard_steps = key_steps = 0
    for seed in range(1000, 1004):
        rng = random.Random(seed)
        env = GridQuest(reward_mode=mode)
        rules = RewardRules.from_layout(env.layout, mode)
        _, info = env.reset(seed=seed)
        for _ in range(1500):
            before = public_state(info)
            action = rng.randrange(4)
            _, reward, terminated, truncated, info = env.step(action)
            assert check_transition_v3(rules, before, action, info, reward) is None
            hazard_steps += env.position in env.hazards
            key_steps += env.position == env.key_at
            if terminated or truncated:
                _, info = env.reset(seed=seed)
    assert hazard_steps and key_steps
