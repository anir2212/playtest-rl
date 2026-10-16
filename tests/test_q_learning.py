from agents.q_learning import train, evaluate


def test_q_learning_is_seed_reproducible():
    qa, ha = train(seed=3, episodes=15)
    qb, hb = train(seed=3, episodes=15)
    assert ha == hb
    assert all((qa[k] == qb[k]).all() for k in qa)
    assert 0 <= evaluate(qa, seed=10, episodes=3)['success_rate'] <= 1
