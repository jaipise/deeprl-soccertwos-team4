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
    parser.add_argument("--eval-results", default="eval_results")
    parser.add_argument("--out-dir", default="report_plots")
    parser.add_argument(
        "--smooth",
        type=int,
        default=5,
        help="Rolling mean window for reward curves.",
    )
    return parser.parse_args()


EXPERIMENT_DIRS = {
    "Baseline": "PPO_selfplay_baseline",
    "Reward Shaping": "PPO_selfplay_shaped",
    "Self-Play Archive": "PPO_selfplay_rec",
    "Curriculum": "PPO_curriculum_multiagent",
    "Curriculum V4": "PPO_curriculum_multiagent_V4_aggressive_combo",
    "Shared V5": "PPO_curriculum_shared_V5_dynamic_aggressive",
}

COLORS = {
    "Baseline": "#5b6472",
    "Reward Shaping": "#0f8b8d",
    "Self-Play Archive": "#7b2cbf",
    "Curriculum": "#c2410c",
    "Curriculum V4": "#dc2626",
    "Shared V5": "#2563eb",
}


def discover_progress_files(ray_results):
    rows = []
    for family, dirname in EXPERIMENT_DIRS.items():
        for path in sorted((ray_results / dirname).glob("*/progress.csv")):
            rows.append({"family": family, "run_id": path.parent.name, "path": path})
    return rows


def load_runs(progress_rows):
    import pandas as pd

    runs = []
    summary = []
    for row in progress_rows:
        path = row["path"]
        try:
            df = pd.read_csv(path)
        except pd.errors.EmptyDataError:
            continue
        if df.empty or "episode_reward_mean" not in df:
            continue
        family = row["family"]
        run_id = row["run_id"]
        label = f"{family} ({run_id.split('_')[1]})"
        runs.append({**row, "label": label, "df": df})
        last = df.iloc[-1]
        best_idx = df["episode_reward_mean"].idxmax()
        best = df.loc[best_idx]
        summary.append(
            {
                "family": family,
                "run_id": run_id,
                "path": str(path),
                "training_iteration": last.get("training_iteration"),
                "timesteps_total": last.get("timesteps_total"),
                "episodes_total": last.get("episodes_total"),
                "episode_reward_mean": last.get("episode_reward_mean"),
                "episode_reward_min": last.get("episode_reward_min"),
                "episode_reward_max": last.get("episode_reward_max"),
                "time_total_s": last.get("time_total_s"),
                "best_reward_mean": best.get("episode_reward_mean"),
                "best_reward_iteration": best.get("training_iteration"),
                "best_reward_timestep": best.get("timesteps_total"),
            }
        )
    return runs, pd.DataFrame(summary)


def select_main_runs(runs):
    selected = {}
    for run in runs:
        family = run["family"]
        last = run["df"].iloc[-1]
        time_total_s = float(last.get("time_total_s", 0.0))
        if family not in selected:
            selected[family] = run
            continue
        prev_time = float(selected[family]["df"].iloc[-1].get("time_total_s", 0.0))
        if time_total_s > prev_time:
            selected[family] = run
    return list(selected.values())


def plot_reward_vs_steps(runs, out_path, smooth):
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(8, 4.8), dpi=180)

    for run in runs:
        label = run["family"]
        df = run["df"]
        x = df["timesteps_total"] if "timesteps_total" in df else df.index
        y = df["episode_reward_mean"]
        if smooth > 1:
            y = y.rolling(smooth, min_periods=1).mean()
        ax.plot(x, y, label=label, linewidth=2.0, color=COLORS.get(label))

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
    for run in runs:
        label = run["family"]
        df = run["df"]
        if "time_total_s" not in df:
            continue
        x = df["time_total_s"] / 3600.0
        y = df["episode_reward_mean"]
        if smooth > 1:
            y = y.rolling(smooth, min_periods=1).mean()
        ax.plot(x, y, label=label, linewidth=2.0, color=COLORS.get(label))

    ax.axhline(0, color="#9ca3af", linewidth=1, linestyle="--")
    ax.set_title("Training Reward by Wall-Clock Time")
    ax.set_xlabel("Hours")
    ax.set_ylabel("Episode reward mean")
    ax.grid(True, color="#e5e7eb", linewidth=0.8)
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)


def plot_all_reward_vs_time(runs, out_path, smooth):
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(9, 5.2), dpi=180)
    seen = set()
    for run in runs:
        label = run["family"]
        df = run["df"]
        if "time_total_s" not in df:
            continue
        x = df["time_total_s"] / 3600.0
        y = df["episode_reward_mean"]
        if smooth > 1:
            y = y.rolling(smooth, min_periods=1).mean()
        ax.plot(
            x,
            y,
            label=label if label not in seen else None,
            linewidth=1.2,
            alpha=0.45,
            color=COLORS.get(label),
        )
        seen.add(label)

    ax.axhline(0, color="#9ca3af", linewidth=1, linestyle="--")
    ax.set_title("All Training Reward Runs by Wall-Clock Time")
    ax.set_xlabel("Hours")
    ax.set_ylabel("Episode reward mean")
    ax.grid(True, color="#e5e7eb", linewidth=0.8)
    ax.legend(frameon=False, ncol=2)
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)


def write_submission_eval_summary(out_dir):
    import pandas as pd

    rows = [
        {
            "agent": "TEAM4_AGENT_REWARD",
            "vs_ceia_wins_out_of_25": 0,
            "ceia_goal_diff_sample": -63,
            "vs_random_score_out_of_25": 25.0,
        },
        {
            "agent": "TEAM4_AGENT_SELFPLAY",
            "vs_ceia_wins_out_of_25": 0,
            "ceia_goal_diff_sample": -113,
            "vs_random_score_out_of_25": 22.222,
        },
        {
            "agent": "TEAM4_AGENT_CURRICULUM_V5",
            "vs_ceia_wins_out_of_25": 0,
            "ceia_goal_diff_sample": -144,
            "vs_random_score_out_of_25": 13.889,
        },
        {
            "agent": "TEAM4_AGENT_SELFPLAY_REC",
            "vs_ceia_wins_out_of_25": 0,
            "ceia_goal_diff_sample": -160,
            "vs_random_score_out_of_25": 25.0,
        },
        {
            "agent": "TEAM4_AGENT_CURRICULUM",
            "vs_ceia_wins_out_of_25": 0,
            "ceia_goal_diff_sample": -162,
            "vs_random_score_out_of_25": 25.0,
        },
        {
            "agent": "TEAM4_AGENT_CURRICULUM_V4",
            "vs_ceia_wins_out_of_25": 0,
            "ceia_goal_diff_sample": -171,
            "vs_random_score_out_of_25": 22.222,
        },
    ]
    df = pd.DataFrame(rows)
    df["rank_by_ceia_goal_diff"] = df["ceia_goal_diff_sample"].rank(
        ascending=False, method="min"
    ).astype(int)
    df = df.sort_values(["vs_ceia_wins_out_of_25", "ceia_goal_diff_sample"], ascending=False)
    df.to_csv(out_dir / "submission_eval_summary.csv", index=False)
    return df


def plot_submission_eval_summary(eval_summary, out_path):
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(8, 3.8), dpi=180)
    plot_df = eval_summary.sort_values("ceia_goal_diff_sample")
    colors = [
        "#0f8b8d" if agent == "TEAM4_AGENT_REWARD" else "#9ca3af"
        for agent in plot_df["agent"]
    ]
    ax.barh(plot_df["agent"], plot_df["ceia_goal_diff_sample"], color=colors)
    ax.axvline(0, color="#111827", linewidth=1)
    ax.set_title("CEIA Goal Differential Separates 0/25 Submissions")
    ax.set_xlabel("Agent goals minus CEIA goals, evaluation sample")
    ax.grid(True, axis="x", color="#e5e7eb", linewidth=0.8)
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)


def read_reward_checkpoint_eval(eval_dir, out_dir):
    import pandas as pd

    rows = []
    for path in sorted(eval_dir.glob("reward_checkpoint_*_vs_ceia.csv")):
        df = pd.read_csv(path)
        games = len(df)
        checkpoint = path.stem.replace("reward_", "").replace("_vs_ceia", "")
        rows.append(
            {
                "checkpoint": checkpoint,
                "games": games,
                "agent_wins": int((df["result"] == "a_win").sum()),
                "ceia_wins": int((df["result"] == "b_win").sum()),
                "draws": int((df["result"] == "draw").sum()),
                "win_rate": float((df["result"] == "a_win").mean()) if games else 0.0,
                "mean_reward_margin": (
                    float((df["a_reward"] - df["b_reward"]).mean()) if games else 0.0
                ),
            }
        )
    result = pd.DataFrame(rows)
    if not result.empty:
        result = result.sort_values("win_rate", ascending=False)
        result.to_csv(out_dir / "reward_checkpoint_eval_summary.csv", index=False)
    return result


def plot_reward_checkpoint_eval(checkpoint_eval, out_path):
    import matplotlib.pyplot as plt

    if checkpoint_eval.empty:
        return
    ordered = checkpoint_eval.sort_values("checkpoint")
    fig, ax = plt.subplots(figsize=(8, 3.8), dpi=180)
    ax.plot(ordered["checkpoint"], ordered["win_rate"], marker="o", color="#0f8b8d")
    ax.set_title("Reward Shaping Checkpoint Selection vs CEIA")
    ax.set_xlabel("Checkpoint")
    ax.set_ylabel("Win rate")
    ax.set_ylim(0, 1)
    ax.grid(True, color="#e5e7eb", linewidth=0.8)
    plt.xticks(rotation=35, ha="right")
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)


def main():
    args = parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    matplotlib.use("Agg")

    progress_rows = discover_progress_files(Path(args.ray_results))
    runs, summary = load_runs(progress_rows)
    if not runs:
        raise SystemExit("No usable progress.csv files found.")
    main_runs = select_main_runs(runs)

    plot_reward_vs_steps(main_runs, out_dir / "reward_vs_timesteps.png", args.smooth)
    plot_reward_vs_time(main_runs, out_dir / "reward_vs_time.png", args.smooth)
    plot_all_reward_vs_time(runs, out_dir / "all_reward_vs_time.png", args.smooth)
    summary.to_csv(out_dir / "training_summary.csv", index=False)
    eval_summary = write_submission_eval_summary(out_dir)
    plot_submission_eval_summary(eval_summary, out_dir / "submission_ceia_goal_diff.png")
    checkpoint_eval = read_reward_checkpoint_eval(Path(args.eval_results), out_dir)
    plot_reward_checkpoint_eval(checkpoint_eval, out_dir / "reward_checkpoint_win_rate.png")
    print(summary.to_string(index=False))
    print("\nSubmission evaluation summary:")
    print(eval_summary.to_string(index=False))
    if not checkpoint_eval.empty:
        print("\nReward checkpoint evaluation summary:")
        print(checkpoint_eval.to_string(index=False))
    print(f"\nWrote plots to {out_dir}")


if __name__ == "__main__":
    main()
