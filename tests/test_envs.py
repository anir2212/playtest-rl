from gymnasium.utils.env_checker import check_env
from envs.gridquest import GridQuest
from envs.slicer import Slicer


def test_gym_conformance():
    check_env(GridQuest())
    check_env(Slicer())


def test_seeded_reset_and_termination():
    a, b = Slicer(), Slicer()
    assert a.reset(seed=9)[0].tolist() == b.reset(seed=9)[0].tolist()
    for _ in range(79):
        assert not a.step(0)[3]
    assert a.step(0)[3]


def test_grid_door_key_and_bug_gap():
    env = GridQuest()
    env.reset(seed=1)
    for _ in range(5):
        _, _, _, _, info = env.step(1)
    assert info['key']
    env = GridQuest(bug='wall_gap')
    env.reset(seed=1)
    env.step(1); env.step(1)
    _, _, done, _, info = env.step(2)
    assert done and info['bug_detected'] == 'wall_gap'


def test_slicer_speed_default_is_unchanged_and_scales_velocities():
    import random
    import pytest
    plain, explicit, faster = Slicer(), Slicer(speed=1.0), Slicer(speed=2.0)
    first = [env.reset(seed=4)[0] for env in (plain, explicit, faster)]
    assert first[0].tolist() == first[1].tolist()
    assert faster.velocity_x == 2.0 * plain.velocity_x
    assert faster.velocity_y == 2.0 * plain.velocity_y
    rng = random.Random(4)
    for _ in range(80):
        action = rng.randrange(4)
        a, b = plain.step(action), explicit.step(action)
        assert a[0].tolist() == b[0].tolist() and a[1:] == b[1:]
    with pytest.raises(ValueError):
        Slicer(speed=0.1)
