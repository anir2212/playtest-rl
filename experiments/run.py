"""Five-seed measured Q, DQN, PPO, SB3 and playtesting runs."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from stable_baselines3 import DQN as SB3DQN, PPO as SB3PPO
from stable_baselines3.common.vec_env import DummyVecEnv

from agents import q_learning, dqn, ppo
from analysis.aggregate import mean_ci
from envs.slicer import Slicer
from playtest.bug_hunt import BUGS, POLICIES, hunt

ROOT = Path(__file__).resolve().parents[1]


def run_q(episodes: int, seeds: int):
    rows = []
    for reward in ("sparse", "shaped", "curiosity", "hackable", "fixed"):
        for seed in range(seeds):
            q, history = q_learning.train(seed, episodes=episodes,
                                          reward_mode="sparse" if reward == "curiosity" else reward,
                                          curiosity_bonus=0.1 if reward == "curiosity" else 0.0)
            evaluation = q_learning.evaluate(q, seed=1000 + seed)
            rows.append({"reward": reward, "seed": seed, "train_last100_success":
                         np.mean([h["success"] for h in history[-100:]]), **evaluation})
    sweeps = []
    for alpha in (0.1, 0.25):
        for gamma in (0.9, 0.99):
            values = []
            for seed in range(seeds):
                q, _ = q_learning.train(seed, episodes=episodes, alpha=alpha, gamma=gamma)
                values.append(q_learning.evaluate(q, seed=2000 + seed)["success_rate"])
            sweeps.append({"alpha": alpha, "gamma": gamma, "success": mean_ci(values)})
    return {"rows": rows, "sweep": sweeps,
            "summary": {reward: mean_ci([r["success_rate"] for r in rows if r["reward"] == reward])
                        for reward in ("sparse", "shaped", "curiosity", "hackable", "fixed")}}


def run_ablations(steps: int, seeds: int):
    """One-factor PPO clip and GAE-lambda ablations, common evaluation seeds."""
    ppo_steps = max(512, steps // 512 * 512)
    rows = []
    for label, clip, lam in (("reference", 0.2, 0.95),
                             ("no_clip", 100.0, 0.95),
                             ("short_gae", 0.2, 0.0)):
        for seed in range(seeds):
            model, _ = ppo.train(seed, steps=ppo_steps, clip=clip, lam=lam)
            rows.append({"setting": label, "seed": seed, "clip": clip,
                         "gae_lambda": lam, "eval": ppo.evaluate(model, seed=1000 + seed)})
    summary = {label: mean_ci([r["eval"]["mean_score"] for r in rows if r["setting"] == label])
               for label in ("reference", "no_clip", "short_gae")}
    return {"rows": rows, "summary_mean_score": summary, "steps": ppo_steps}


def eval_sb3(model, seed: int, episodes: int = 30):
    env = Slicer()
    scores, bombs, returns = [], [], []
    for ep in range(episodes):
        obs, _ = env.reset(seed=seed * 100000 + ep)
        total = 0
        for _ in range(env.horizon):
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = env.step(int(action))
            total += reward
            if terminated or truncated:
                break
        scores.append(info["score"]); bombs.append(info["bomb_hits"]); returns.append(total)
    env.close()
    return {"mean_score": float(np.mean(scores)), "mean_bomb_hits": float(np.mean(bombs)),
            "mean_return": float(np.mean(returns))}


def run_neural(steps: int, seeds: int):
    rows = []
    ppo_steps = max(512, steps // 512 * 512)
    for seed in range(seeds):
        model, log = dqn.train(seed, steps=steps)
        rows.append({"agent": "double_dqn", "seed": seed, "train": log,
                     "eval": dqn.evaluate(model, seed=1000 + seed)})
        model, log = ppo.train(seed, steps=ppo_steps)
        rows.append({"agent": "scratch_ppo", "seed": seed, "train": log,
                     "eval": ppo.evaluate(model, seed=1000 + seed)})
        torch.manual_seed(seed)
        sb_env = Slicer()
        sb_dqn = SB3DQN("MlpPolicy", sb_env, seed=seed, verbose=0, device="cpu",
                        learning_rate=5e-4, buffer_size=10000, learning_starts=100,
                        batch_size=64, train_freq=4, gradient_steps=1, target_update_interval=250,
                        exploration_fraction=0.7, exploration_final_eps=0.05, gamma=0.99)
        sb_dqn.learn(total_timesteps=steps)
        rows.append({"agent": "sb3_dqn", "seed": seed, "train": {"steps": steps},
                     "eval": eval_sb3(sb_dqn, seed=1000 + seed)})
        sb_env.close()
        sb_vec = DummyVecEnv([lambda: Slicer() for _ in range(4)])
        sb_ppo = SB3PPO("MlpPolicy", sb_vec, seed=seed, verbose=0, device="cpu",
                        learning_rate=3e-4, n_steps=128, batch_size=128, n_epochs=4,
                        gamma=0.99, gae_lambda=0.95, clip_range=0.2, ent_coef=0.01)
        sb_ppo.learn(total_timesteps=ppo_steps)
        rows.append({"agent": "sb3_ppo", "seed": seed, "train": {"steps": ppo_steps},
                     "eval": eval_sb3(sb_ppo, seed=1000 + seed)})
        sb_vec.close()
    summary = {name: {metric: mean_ci([r["eval"][metric] for r in rows if r["agent"] == name])
                      for metric in ("mean_score", "mean_bomb_hits", "mean_return")}
               for name in ("double_dqn", "scratch_ppo", "sb3_dqn", "sb3_ppo")}
    return {"rows": rows, "summary": summary, "steps": steps, "ppo_steps": ppo_steps}


def run_playtest(seeds: int, max_steps: int):
    rows = [{"bug": bug, "policy": policy, "seed": seed,
             **hunt(bug, policy, seed, max_steps=max_steps)}
            for bug in BUGS for policy in POLICIES for seed in range(seeds)]
    summary = []
    for bug in BUGS:
        for policy in POLICIES:
            group = [r for r in rows if r["bug"] == bug and r["policy"] == policy]
            discovered = [r["time_to_discovery"] for r in group if r["time_to_discovery"] is not None]
            summary.append({"bug": bug, "policy": policy, "discovered": len(discovered),
                            "seeds": seeds, "time_to_discovery_detected_only": mean_ci(discovered) if discovered else None,
                            "mean_position_coverage": float(np.mean([r["position_coverage"] for r in group]))})
    return {"rows": rows, "summary": summary, "censoring_limit_steps": max_steps}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--suite", choices=("q", "neural", "playtest", "ablations"), required=True)
    parser.add_argument("--seeds", type=int, default=5)
    parser.add_argument("--episodes", type=int, default=500)
    parser.add_argument("--steps", type=int, default=5120)
    parser.add_argument("--max-hunt-steps", type=int, default=1200)
    args = parser.parse_args()
    if args.seeds < 5:
        print("Warning: smoke only; plan calls for at least five seeds.")
    result = (run_q(args.episodes, args.seeds) if args.suite == "q" else
              run_neural(args.steps, args.seeds) if args.suite == "neural" else
              run_ablations(args.steps, args.seeds) if args.suite == "ablations" else
              run_playtest(args.seeds, args.max_hunt_steps))
    output = ROOT / "runs" / f"{args.suite}.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2))
    print(output, 'written')
    if args.suite == 'neural':
        print(json.dumps({k: v['mean_score'] for k, v in result['summary'].items()}, indent=2))
    elif args.suite == 'q':
        print(json.dumps(result['summary'], indent=2))
    elif args.suite == "playtest":
        print(json.dumps(result['summary'], indent=2))
    else:
        print(json.dumps(result['summary_mean_score'], indent=2))


if __name__ == "__main__":
    main()
