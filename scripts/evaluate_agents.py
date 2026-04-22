#!/usr/bin/env python
"""Evaluate Soccer Twos agents by playing headless matches.

Examples:
    python scripts/evaluate_agents.py \
        --agent-a TEAM4_AGENT_CURRICULUM \
        --agent-b ceia_baseline_agent \
        --episodes 20 --swap-sides

    python scripts/evaluate_agents.py \
        --agent-a-checkpoint ray_results/PPO_curriculum_multiagent_V4_aggressive_combo \
        --agent-a-name V4_aggressive_combo \
        --agent-b ceia_baseline_agent \
        --episodes 20 --swap-sides --csv eval_results/v4_vs_ceia.csv
"""
import argparse
import csv
import importlib
import os
import pickle
import re
import shutil
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_TEMPLATE = ROOT / "submission" / "TEAM4_AGENT_CURRICULUM"
DEFAULT_EVAL_DIR = ROOT / ".eval_agents"
TEAM0_IDS = (0, 1)
TEAM1_IDS = (2, 3)


class RandomTeamAgent:
    def __init__(self, env):
        self.name = "random"
        self.action_space = env.action_space

    def act(self, observation):
        return {pid: self.action_space.sample() for pid in observation}


@dataclass
class MatchResult:
    episode: int
    side: str
    a_reward: float
    b_reward: float
    reward_margin: float
    steps: int
    result: str


def parse_args():
    parser = argparse.ArgumentParser(
        description="Play Soccer Twos matches and report win/loss/draw stats."
    )
    parser.add_argument("--agent-a", default=None, help="Agent A module/path or 'random'.")
    parser.add_argument("--agent-b", required=True, help="Agent B module/path or 'random'.")
    parser.add_argument(
        "--agent-a-checkpoint",
        default=None,
        help="RLlib checkpoint file or run directory to package and evaluate as Agent A.",
    )
    parser.add_argument(
        "--agent-b-checkpoint",
        default=None,
        help="RLlib checkpoint file or run directory to package and evaluate as Agent B.",
    )
    parser.add_argument("--agent-a-name", default="agent_a")
    parser.add_argument("--agent-b-name", default="agent_b")
    parser.add_argument(
        "--checkpoint-template",
        default=str(DEFAULT_TEMPLATE),
        help="Pure-torch curriculum agent template used for packaged checkpoints.",
    )
    parser.add_argument(
        "--eval-dir",
        default=str(DEFAULT_EVAL_DIR),
        help="Directory for temporary packaged checkpoint agents.",
    )
    parser.add_argument(
        "--episodes",
        type=int,
        default=20,
        help="Episodes per matchup side. With --swap-sides, total games doubles.",
    )
    parser.add_argument("--max-steps", type=int, default=1000)
    parser.add_argument("--swap-sides", action="store_true")
    parser.add_argument("--render", action="store_true")
    parser.add_argument("--csv", default=None, help="Optional per-episode CSV output.")
    parser.add_argument("--seed", type=int, default=None)
    return parser.parse_args()


def resolve_checkpoint_path(path):
    path = Path(path)
    if path.is_file():
        return path
    if not path.is_dir():
        raise FileNotFoundError(f"Checkpoint path not found: {path}")

    candidates = []
    for candidate in path.rglob("checkpoint-*"):
        if candidate.is_file() and not candidate.name.endswith(".tune_metadata"):
            candidates.append(candidate)
    if not candidates:
        raise FileNotFoundError(f"No checkpoint-* files found under {path}")

    def checkpoint_key(candidate):
        match = re.search(r"checkpoint-(\d+)$", candidate.name)
        number = int(match.group(1)) if match else -1
        return number, candidate.stat().st_mtime

    return sorted(candidates, key=checkpoint_key)[-1]


def extract_policy_weights(checkpoint_path, out_dir, policy_ids=("striker", "goalie")):
    import ray.rllib  # noqa: F401  Required so pickle can resolve RLlib classes.
    import torch

    with open(checkpoint_path, "rb") as f:
        trainer_state = pickle.load(f)
    worker_state = pickle.loads(trainer_state["worker"])
    policy_states = worker_state["state"]

    out_dir.mkdir(parents=True, exist_ok=True)
    for policy_id in policy_ids:
        if policy_id not in policy_states:
            raise KeyError(
                f"Policy '{policy_id}' missing from checkpoint {checkpoint_path}"
            )
        state = policy_states[policy_id]
        tensor_state = {
            key: torch.from_numpy(np.asarray(value))
            for key, value in state.items()
            if isinstance(value, np.ndarray) and key != "_optimizer_variables"
        }
        torch.save(tensor_state, out_dir / f"{policy_id}.pth")


def copy_checkpoint_template(template_dir, out_dir):
    template_dir = Path(template_dir)
    if not (template_dir / "agent.py").is_file():
        raise FileNotFoundError(f"Missing checkpoint template agent.py: {template_dir}")
    out_dir.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(template_dir / "agent.py", out_dir / "agent.py")
    init_path = template_dir / "__init__.py"
    if init_path.is_file():
        shutil.copyfile(init_path, out_dir / "__init__.py")
    else:
        (out_dir / "__init__.py").write_text("from .agent import TeamAgent\n")


def package_checkpoint(checkpoint, name, template_dir, eval_dir):
    checkpoint_path = resolve_checkpoint_path(checkpoint)
    safe_name = re.sub(r"[^A-Za-z0-9_]+", "_", name).strip("_") or "checkpoint_agent"
    out_dir = Path(eval_dir) / safe_name
    if out_dir.exists():
        shutil.rmtree(out_dir)
    copy_checkpoint_template(template_dir, out_dir)
    extract_policy_weights(checkpoint_path, out_dir)
    print(f"[package] {name}: {checkpoint_path} -> {out_dir}")
    return str(out_dir)


def import_agent_class(agent_ref):
    if agent_ref == "random":
        return RandomTeamAgent

    path = Path(agent_ref)
    if path.exists():
        path = path.resolve()
        if path.is_file():
            module_dir = path.parent
            module_name = path.stem
        else:
            module_dir = path.parent
            module_name = path.name
        sys.path.insert(0, str(module_dir))
    else:
        module_name = agent_ref

    try:
        module = importlib.import_module(module_name)
    except Exception as exc:
        raise RuntimeError(f"Could not import agent module '{agent_ref}': {exc}") from exc

    for class_name in ("TeamAgent", "RayAgent", "RandomAgent"):
        if hasattr(module, class_name):
            return getattr(module, class_name)
    raise AttributeError(
        f"Module '{agent_ref}' does not export TeamAgent, RayAgent, or RandomAgent"
    )


def make_env(render=False):
    import soccer_twos

    return soccer_twos.make(render=render)


def done_any(done):
    if isinstance(done, dict):
        return any(bool(value) for value in done.values())
    return bool(done)


def reset_env(env):
    reset_result = env.reset()
    if isinstance(reset_result, tuple) and len(reset_result) == 2:
        return reset_result[0]
    return reset_result


def reward_get(reward, player_id):
    return float(reward.get(player_id, reward.get(str(player_id), 0.0)))


def subset_obs(obs, player_ids):
    return {pid: obs[pid] for pid in player_ids if pid in obs}


def play_episode(env, team0_agent, team1_agent, max_steps):
    obs = reset_env(env)
    team0_reward = 0.0
    team1_reward = 0.0

    for step in range(1, max_steps + 1):
        action = {}
        action.update(team0_agent.act(subset_obs(obs, TEAM0_IDS)))
        action.update(team1_agent.act(subset_obs(obs, TEAM1_IDS)))
        obs, reward, done, _info = env.step(action)
        team0_reward += sum(reward_get(reward, pid) for pid in TEAM0_IDS)
        team1_reward += sum(reward_get(reward, pid) for pid in TEAM1_IDS)
        if done_any(done):
            break

    if team0_reward > team1_reward:
        winner = "team0"
    elif team1_reward > team0_reward:
        winner = "team1"
    else:
        winner = "draw"
    return team0_reward, team1_reward, step, winner


def evaluate_side(env, agent_a_cls, agent_b_cls, episodes, max_steps, a_on_team0):
    team0_agent = agent_a_cls(env) if a_on_team0 else agent_b_cls(env)
    team1_agent = agent_b_cls(env) if a_on_team0 else agent_a_cls(env)
    side = "a_team0" if a_on_team0 else "a_team1"
    results = []

    for episode in range(1, episodes + 1):
        team0_reward, team1_reward, steps, winner = play_episode(
            env, team0_agent, team1_agent, max_steps
        )
        if a_on_team0:
            a_reward, b_reward = team0_reward, team1_reward
            a_winner = winner == "team0"
            b_winner = winner == "team1"
        else:
            a_reward, b_reward = team1_reward, team0_reward
            a_winner = winner == "team1"
            b_winner = winner == "team0"

        if a_winner:
            result = "a_win"
        elif b_winner:
            result = "b_win"
        else:
            result = "draw"

        results.append(
            MatchResult(
                episode=episode,
                side=side,
                a_reward=a_reward,
                b_reward=b_reward,
                reward_margin=a_reward - b_reward,
                steps=steps,
                result=result,
            )
        )
        print(
            f"[{side} ep {episode:03d}] {result:6s} "
            f"a_reward={a_reward:.3f} b_reward={b_reward:.3f} steps={steps}"
        )
    return results


def summarize(results, agent_a_name, agent_b_name):
    total = len(results)
    a_wins = sum(result.result == "a_win" for result in results)
    b_wins = sum(result.result == "b_win" for result in results)
    draws = sum(result.result == "draw" for result in results)
    margins = np.asarray([result.reward_margin for result in results], dtype=np.float64)
    steps = np.asarray([result.steps for result in results], dtype=np.float64)

    print("\nSummary")
    print(f"  {agent_a_name} vs {agent_b_name}")
    print(f"  games: {total}")
    print(
        f"  {agent_a_name}: {a_wins} wins ({a_wins / total:.1%}), "
        f"{agent_b_name}: {b_wins} wins ({b_wins / total:.1%}), "
        f"draws: {draws} ({draws / total:.1%})"
    )
    print(f"  mean reward margin ({agent_a_name} - {agent_b_name}): {margins.mean():.3f}")
    print(f"  median reward margin: {np.median(margins):.3f}")
    print(f"  mean episode steps: {steps.mean():.1f}")


def write_csv(path, results):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "episode",
                "side",
                "a_reward",
                "b_reward",
                "reward_margin",
                "steps",
                "result",
            ],
        )
        writer.writeheader()
        for result in results:
            writer.writerow(result.__dict__)
    print(f"[csv] wrote {path}")


def main():
    args = parse_args()
    if args.seed is not None:
        np.random.seed(args.seed)

    agent_a_ref = args.agent_a
    agent_b_ref = args.agent_b
    if args.agent_a_checkpoint:
        agent_a_ref = package_checkpoint(
            args.agent_a_checkpoint,
            args.agent_a_name,
            args.checkpoint_template,
            args.eval_dir,
        )
    if args.agent_b_checkpoint:
        agent_b_ref = package_checkpoint(
            args.agent_b_checkpoint,
            args.agent_b_name,
            args.checkpoint_template,
            args.eval_dir,
        )
    if not agent_a_ref:
        raise ValueError("Provide --agent-a or --agent-a-checkpoint")

    os.chdir(ROOT)
    sys.path.insert(0, str(ROOT))
    agent_a_cls = import_agent_class(agent_a_ref)
    agent_b_cls = import_agent_class(agent_b_ref)

    env = make_env(render=args.render)
    try:
        results = evaluate_side(
            env,
            agent_a_cls,
            agent_b_cls,
            args.episodes,
            args.max_steps,
            a_on_team0=True,
        )
        if args.swap_sides:
            results.extend(
                evaluate_side(
                    env,
                    agent_a_cls,
                    agent_b_cls,
                    args.episodes,
                    args.max_steps,
                    a_on_team0=False,
                )
            )
    finally:
        if hasattr(env, "close"):
            env.close()

    summarize(results, args.agent_a_name, args.agent_b_name)
    if args.csv:
        write_csv(args.csv, results)


if __name__ == "__main__":
    main()
