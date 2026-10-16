import unittest
from playtestrl import GridQuest, bug_hunt, train


class RLTests(unittest.TestCase):
    def test_reset_and_bounds(self):
        env = GridQuest()
        self.assertEqual(env.reset(7), env.reset(7))
        env.step(2)
        self.assertEqual(env.position, (0, 0))
        with self.assertRaises(ValueError):
            env.step(9)

    def test_learning_and_bug(self):
        q, rate = train(episodes=10)
        self.assertTrue(q)
        self.assertTrue(0 <= rate <= 1)
        self.assertEqual(bug_hunt(seed=3), bug_hunt(seed=3))


if __name__ == "__main__":
    unittest.main()
