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
