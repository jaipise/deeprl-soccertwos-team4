import argparse
import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "report_plots/.matplotlib")

import matplotlib


def parse_args():
    parser = argparse.ArgumentParser(
        description="Generate report plots from Ray RLlib progress.csv files."
    )
    parser.add_argument("--ray-results", default="ray_results")
    parser.add_argument("--out-dir", default="report_plots")
    parser.add_argument(
        "--smooth",
        type=int,
        default=5,
        help="Rolling mean window for reward curves.",
    )
    return parser.parse_args()


def latest_progress_files(ray_results):
    candidates = {
        "Baseline": "PPO_selfplay_baseline",
        "Reward Shaping": "PPO_selfplay_shaped",
        "Self-Play Archive": "PPO_selfplay_rec",
        "Curriculum": "PPO_curriculum_multiagent",
    }
    selected = {}
    for label, dirname in candidates.items():
        paths = list((ray_results / dirname).glob("*/progress.csv"))
        if not paths:
            continue
        selected[label] = max(paths, key=lambda p: p.stat().st_mtime)
    return selected


def load_runs(progress_files):
    import pandas as pd

    runs = {}
    summary = []
    for label, path in progress_files.items():
        df = pd.read_csv(path)
        if df.empty or "episode_reward_mean" not in df:
            continue
        runs[label] = df
        last = df.iloc[-1]
        summary.append(
            {
                "agent": label,
                "path": str(path),
                "training_iteration": last.get("training_iteration"),
                "timesteps_total": last.get("timesteps_total"),
                "episodes_total": last.get("episodes_total"),
                "episode_reward_mean": last.get("episode_reward_mean"),
                "episode_reward_min": last.get("episode_reward_min"),
                "episode_reward_max": last.get("episode_reward_max"),
                "time_total_s": last.get("time_total_s"),
            }
        )
    return runs, pd.DataFrame(summary)


def plot_reward_vs_steps(runs, out_path, smooth):
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(8, 4.8), dpi=180)
    colors = {
        "Baseline": "#5b6472",
        "Reward Shaping": "#0f8b8d",
        "Self-Play Archive": "#7b2cbf",
        "Curriculum": "#c2410c",
    }

    for label, df in runs.items():
        x = df["timesteps_total"] if "timesteps_total" in df else df.index
        y = df["episode_reward_mean"]
        if smooth > 1:
            y = y.rolling(smooth, min_periods=1).mean()
        ax.plot(x, y, label=label, linewidth=2.0, color=colors.get(label))

    ax.axhline(0, color="#9ca3af", linewidth=1, linestyle="--")
    ax.set_title("Training Reward by Environment Steps")
    ax.set_xlabel("Timesteps")
    ax.set_ylabel("Episode reward mean")
    ax.grid(True, color="#e5e7eb", linewidth=0.8)
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)


def plot_reward_vs_time(runs, out_path, smooth):
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(8, 4.8), dpi=180)
    for label, df in runs.items():
        if "time_total_s" not in df:
            continue
        x = df["time_total_s"] / 3600.0
        y = df["episode_reward_mean"]
        if smooth > 1:
            y = y.rolling(smooth, min_periods=1).mean()
        ax.plot(x, y, label=label, linewidth=2.0)

    ax.axhline(0, color="#9ca3af", linewidth=1, linestyle="--")
    ax.set_title("Training Reward by Wall-Clock Time")
    ax.set_xlabel("Hours")
    ax.set_ylabel("Episode reward mean")
    ax.grid(True, color="#e5e7eb", linewidth=0.8)
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)


def main():
    args = parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    matplotlib.use("Agg")

    progress_files = latest_progress_files(Path(args.ray_results))
    runs, summary = load_runs(progress_files)
    if not runs:
        raise SystemExit("No usable progress.csv files found.")

    plot_reward_vs_steps(runs, out_dir / "reward_vs_timesteps.png", args.smooth)
    plot_reward_vs_time(runs, out_dir / "reward_vs_time.png", args.smooth)
    summary.to_csv(out_dir / "training_summary.csv", index=False)
    print(summary.to_string(index=False))
    print(f"\nWrote plots to {out_dir}")


if __name__ == "__main__":
    main()
