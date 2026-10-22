"""Rebuild evidence-linked learning and exploration figures."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]


def curve_figure(evidence: Path, output: Path):
    """Plot checkpoint score against environment steps with seed-level 95% intervals."""
    data = json.loads(evidence.read_text())
    if not data.get("matches_protocol"):
        raise ValueError(f"{evidence} does not match the predeclared curve protocol")
    fig, axis = plt.subplots(figsize=(7.5, 4.2), constrained_layout=True)
    colors = {"double_dqn": "#247ba0", "scratch_ppo": "#e4572e"}
    for agent, summary in data["summary"].items():
        steps = [c["step"] for c in summary["curve"]]
        means = [c["mean_score"]["mean"] for c in summary["curve"]]
        low = [c["mean_score"]["ci95"][0] for c in summary["curve"]]
        high = [c["mean_score"]["ci95"][1] for c in summary["curve"]]
        label = f"{agent.replace('_', ' ')} ({summary['seeds']} seeds)"
        axis.plot(steps, means, marker="o", color=colors.get(agent), label=label)
        axis.fill_between(steps, low, high, color=colors.get(agent), alpha=0.18)
    axis.axhline(data["protocol"]["threshold"], color="grey", linestyle="--", linewidth=1,
                 label=f"predeclared threshold {data['protocol']['threshold']:g}")
    axis.set_xlabel("Environment interactions (training steps)")
    axis.set_ylabel("Slicer fruit score / 80-step episode")
    axis.set_title("Checkpoint learning curves; band = 95% bootstrap over seed means")
    axis.legend(loc="upper left", fontsize=8)
    source = evidence.relative_to(ROOT) if evidence.is_relative_to(ROOT) else evidence
    axis.text(0.99, 0.02, f"source: {source}", transform=axis.transAxes,
              ha="right", va="bottom", fontsize=7, color="grey")
    fig.savefig(output, dpi=180)
    plt.close(fig)


def difficulty_figure(evidence: Path, output: Path):
    """Dot plot of final Slicer scores per difficulty setting, one panel per agent."""
    data = json.loads(evidence.read_text())
    if data.get("protocol") != "slicer-difficulty-v1" or not data.get("curve_consistency", {}).get("identical"):
        raise ValueError(f"{evidence} is not a consistent slicer-difficulty-v1 run")
    summary = data["summary"]
    settings = list(summary["anchors"])
    rows = list(range(len(settings)))[::-1]
    fig, axes = plt.subplots(1, len(summary["agents"]), figsize=(10, 4.9), sharey=True,
                             sharex=True, constrained_layout=True)
    for axis, (agent, info) in zip(axes, summary["agents"].items()):
        for y, setting in zip(rows, settings):
            anchors = summary["anchors"][setting]
            axis.plot(anchors["random"]["mean_score"]["mean"], y, marker="|", markersize=12,
                      color="#9a9a94", linestyle="none")
            axis.plot(anchors["scripted"]["mean_score"]["mean"], y, marker="x", markersize=8,
                      color="#5f5e5a", linestyle="none")
            entry = info["settings"][setting]
            ci = entry["mean_score"]
            axis.errorbar(ci["mean"], y + 0.12, xerr=[[ci["mean"] - ci["ci95"][0]],
                                                      [ci["ci95"][1] - ci["mean"]]],
                          fmt="o", markersize=7, color="#2a78d6", elinewidth=2, capsize=0)
            if "in_setting_reference" in entry:
                ref = entry["in_setting_reference"]["mean_score"]
                axis.errorbar(ref["mean"], y - 0.12, xerr=[[ref["mean"] - ref["ci95"][0]],
                                                           [ref["ci95"][1] - ref["mean"]]],
                              fmt="s", markersize=6, color="#eb6834", elinewidth=2, capsize=0)
        axis.set_title(agent.replace("_", " "), fontsize=10)
        axis.set_xlabel("Fruit score / 80-step episode at 20,480 steps")
        axis.grid(axis="x", color="#e4e3dd", linewidth=0.8)
        axis.set_axisbelow(True)
        for side in ("top", "right"):
            axis.spines[side].set_visible(False)
    axes[0].set_yticks(rows, [s + (" (training)" if s == "train" else "") for s in settings])
    handles = [plt.Line2D([], [], marker="o", color="#2a78d6", linestyle="none",
                          label="trained on default setting (5 seeds, 95% CI)"),
               plt.Line2D([], [], marker="s", color="#eb6834", linestyle="none",
                          label="trained on this setting (5 seeds, 95% CI)"),
               plt.Line2D([], [], marker="x", color="#5f5e5a", linestyle="none", label="scripted anchor"),
               plt.Line2D([], [], marker="|", markersize=12, color="#9a9a94", linestyle="none",
                          label="random anchor")]
    fig.legend(handles=handles, loc="outside lower center", ncol=4, fontsize=8, frameon=False)
    source = evidence.relative_to(ROOT) if evidence.is_relative_to(ROOT) else evidence
    fig.suptitle("Slicer held-out difficulty settings (synthetic game); fruit score only, "
                 f"bomb hits are in RESULTS.md\nsource: {source}", fontsize=9)
    fig.savefig(output, dpi=180)
    plt.close(fig)


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--curve", type=Path, help="draw only the checkpoint curve from this JSON")
    parser.add_argument("--difficulty", type=Path, help="draw only the difficulty figure from this JSON")
    args = parser.parse_args(argv)
    if args.difficulty:
        path = args.difficulty if args.difficulty.is_absolute() else ROOT / args.difficulty
        (ROOT / "figures").mkdir(exist_ok=True)
        difficulty_figure(path, ROOT / "figures" / "slicer_difficulty.png")
        print("difficulty figure written")
        return
    if args.curve:
        path = args.curve if args.curve.is_absolute() else ROOT / args.curve
        (ROOT / "figures").mkdir(exist_ok=True)
        curve_figure(path, ROOT / "figures" / "learning_curve.png")
        print("curve figure written")
        return
    runs, figures = ROOT / "runs", ROOT / "figures"
    figures.mkdir(exist_ok=True)
    q = json.loads((runs / "q.json").read_text())
    neural = json.loads((runs / "neural.json").read_text())
    playtest = json.loads((runs / "playtest.json").read_text())
    fig, axes = plt.subplots(1, 2, figsize=(11, 4), constrained_layout=True)
    modes = list(q["summary"])
    scores = [q["summary"][m]["mean"] for m in modes]
    axes[0].bar(modes, scores, color=["#247ba0", "#70c1b3", "#f3a712", "#e4572e", "#70c1b3"])
    axes[0].set_ylim(0, 1.1)
    axes[0].set_ylabel("Held-out episode success rate")
    axes[0].set_title("GridQuest Q-learning, five training seeds")
    agents = list(neural["summary"])
    means = [neural["summary"][a]["mean_score"]["mean"] for a in agents]
    low = [means[i] - neural["summary"][a]["mean_score"]["ci95"][0] for i, a in enumerate(agents)]
    high = [neural["summary"][a]["mean_score"]["ci95"][1] - means[i] for i, a in enumerate(agents)]
    axes[1].bar(range(len(agents)), means, color="#247ba0")
    axes[1].errorbar(range(len(agents)), means, yerr=[low, high], fmt="none", ecolor="black", capsize=3)
    axes[1].set_xticks(range(len(agents)), [a.replace("_", "\n") for a in agents])
    axes[1].set_ylabel("Slicer fruit score / 80-step episode")
    axes[1].set_title("5,120 training steps, 30 evaluation episodes/seed")
    fig.savefig(figures / "learning_comparison.png", dpi=180)
    plt.close(fig)

    fig, axes = plt.subplots(1, 3, figsize=(11, 3.7), constrained_layout=True)
    for axis, policy in zip(axes, ("random", "coverage", "q_count")):
        row = next(r for r in playtest["rows"] if r["bug"] == "softlock" and
                   r["policy"] == policy and r["seed"] == 0)
        data = np.asarray(row["position_visits"])
        axis.imshow(np.log1p(data), cmap="viridis", vmin=0, vmax=np.log1p(1200))
        axis.set_title(f"{policy}: {row['position_coverage']:.0%} positions")
        axis.set_xlabel("x")
        axis.set_ylabel("y")
    fig.suptitle("Softlock search: log(1 + visits), seed 0, censored at discovery")
    fig.savefig(figures / "coverage_heatmaps.png", dpi=180)
    plt.close(fig)
    print("figures written")


if __name__ == "__main__":
    main()
