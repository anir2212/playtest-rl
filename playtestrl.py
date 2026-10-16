"""A small grid game, tabular Q-learning and coverage-driven playtesting."""
import argparse
import json
import random
from collections import defaultdict

ACTIONS = ((0, -1), (0, 1), (-1, 0), (1, 0))


class GridQuest:
    def __init__(self, size=6, bug_gap=False):
        if size < 5:
            raise ValueError("size must be at least 5")
        self.size, self.bug_gap = size, bug_gap
        self.reset()

    def reset(self, seed=None):
        self.rng = random.Random(seed)
        self.position = (0, 0)
        self.key = False
        self.steps = 0
        return self.observe()

    def observe(self):
        return (*self.position, self.key)

    def step(self, action):
        if action not in range(4):
            raise ValueError("action must be 0..3")
        x, y = self.position
        dx, dy = ACTIONS[action]
        nx, ny = x + dx, y + dy
        if self.bug_gap and (nx, ny) == (-1, 2):
            self.position = (nx, ny)
        elif 0 <= nx < self.size and 0 <= ny < self.size and (nx, ny) != (2, 2):
            self.position = (nx, ny)
        self.steps += 1
        if self.position == (0, self.size-1):
            self.key = True
        success = self.position == (self.size-1, self.size-1) and self.key
        done = success or self.steps >= self.size*self.size*4
        reward = 1.0 if success else -0.01
        return self.observe(), reward, done, {"success": success, "bug": not (0 <= self.position[0] < self.size and 0 <= self.position[1] < self.size)}


def train(seed=7, episodes=1000, size=6, shaped=True):
    rng = random.Random(seed)
    env = GridQuest(size)
    q = defaultdict(lambda: [0.0]*4)
    successes = 0
    for ep in range(episodes):
        state = env.reset(seed+ep)
        epsilon = max(.05, .9*(1-ep/episodes))
        for _ in range(size*size*4):
            action = rng.randrange(4) if rng.random() < epsilon else max(range(4), key=lambda a: q[state][a])
            new, reward, done, info = env.step(action)
            if shaped and new[2] and not state[2]:
                reward += .2
            q[state][action] += .2*(reward + (.95*max(q[new]) if not done else 0) - q[state][action])
            state = new
            if done:
                successes += int(info["success"])
                break
    return q, successes/episodes


def evaluate(q, seed=100, episodes=100, size=6):
    env = GridQuest(size)
    wins, visited = 0, set()
    for ep in range(episodes):
        state = env.reset(seed+ep)
        for _ in range(size*size*4):
            visited.add(state)
            state, _, done, info = env.step(max(range(4), key=lambda a: q[state][a]))
            if done:
                wins += int(info["success"])
                break
    return {"success_rate": wins/episodes, "visited_states": len(visited)}


def bug_hunt(policy="random", seed=7, max_steps=1000):
    rng = random.Random(seed)
    env = GridQuest(bug_gap=True)
    env.reset(seed)
    seen = defaultdict(int)
    tried = defaultdict(int)
    for step in range(1, max_steps+1):
        state = env.observe()
        if policy == "coverage":
            choices = []
            for action, (dx, dy) in enumerate(ACTIONS):
                next_pos = (state[0]+dx, state[1]+dy)
                choices.append((tried[(state, action)], seen[next_pos], rng.random(), action))
            action = min(choices)[3]
        else:
            action = rng.randrange(4)
        tried[(state, action)] += 1
        state, _, done, info = env.step(action)
        seen[state[:2]] += 1
        if info["bug"]:
            return step
        if done:
            env.reset(seed+step)
    return None


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--episodes", type=int, default=1000)
    args = p.parse_args()
    runs = []
    for seed in range(5):
        q, training_success = train(seed, args.episodes)
        runs.append({"seed": seed, "train_success": training_success, **evaluate(q, seed+1000), "bug_random_steps": bug_hunt("random", seed), "bug_coverage_steps": bug_hunt("coverage", seed)})
    print(json.dumps(runs, indent=2))


if __name__ == "__main__":
    main()
