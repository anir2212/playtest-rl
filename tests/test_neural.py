import random
import numpy as np
import torch
from agents.dqn import ReplayBuffer, QNetwork, double_dqn_target
from agents.ppo import gae_advantages, ActorCritic


def test_replay_copies_and_samples():
    buffer = ReplayBuffer(3)
    obs = np.ones(6, dtype=np.float32)
    buffer.add(obs, 2, 1.0, obs * 2, False)
    obs[:] = 9
    assert buffer.sample(1, random.Random(1))[0][0, 0].item() == 1


def test_target_and_double_dqn_terminal():
    online, target = QNetwork(), QNetwork()
    target.load_state_dict(online.state_dict())
    assert all(torch.equal(a, b) for a, b in zip(online.parameters(), target.parameters()))
    values = double_dqn_target(online, target, torch.zeros(2, 6),
                               torch.tensor([1.0, 2.0]), torch.ones(2), 0.99)
    assert values.tolist() == [1.0, 2.0]


def test_gae_stops_at_done_and_actor_shapes():
    a, r = gae_advantages(np.array([[1.0], [2.0]], np.float32),
                           np.zeros((2, 1), np.float32),
                           np.array([[1.0], [1.0]], np.float32), np.zeros(1))
    assert a[:, 0].tolist() == [1.0, 2.0]
    policy, value = ActorCritic()(torch.zeros(2, 6))
    assert policy.probs.shape == (2, 4) and value.shape == (2,)
