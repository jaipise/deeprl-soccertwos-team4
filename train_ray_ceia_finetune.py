import argparse
import csv
import json
import os
import pickle
import time
from pathlib import Path

import numpy as np
import ray
from ray import tune
from ray.tune.registry import get_trainable_cls
import torch

from utils import create_rllib_env


NUM_ENVS_PER_WORKER = 1
POLICY_KEYS = {
    "_hidden_layers.0._model.0.weight",
    "_hidden_layers.0._model.0.bias",
    "_hidden_layers.1._model.0.weight",
    "_hidden_layers.1._model.0.bias",
    "_logits._model.0.weight",
    "_logits._model.0.bias",
    "_value_branch._model.0.weight",
    "_value_branch._model.0.bias",
}
ACTION_POLICY_KEYS = {
    "_hidden_layers.0._model.0.weight",
    "_hidden_layers.0._model.0.bias",
    "_hidden_layers.1._model.0.weight",
    "_hidden_layers.1._model.0.bias",
    "_logits._model.0.weight",
    "_logits._model.0.bias",
}

CEIA_PROB = 0.85

# Make the CEIA fine-tune play forward instead of camping in the shaped-reward
# equilibrium. This stays local to the CEIA script.
CEIA_FINETUNE_SHAPING_OVERRIDES = {
    "time_penalty": 0.003,
    "score_bonus": 2.0,
    "concede_penalty": 0.5,
    "progress_coef": 0.008,
    "proximity_coef": 0.002,
    "spacing_good_bonus": 0.00025,
    "spacing_close_penalty": 0.00025,
    "goalie_target_coef": 0.0005,
    "goalie_block_coef": 0.00025,
    "striker_support_coef": 0.0025,
    "striker_support_offset": 1.5,
    "role_mode": "dynamic",
    "touch_bonus": 0.05,
}


def _shaping_overrides_from_env():
    params = dict(CEIA_FINETUNE_SHAPING_OVERRIDES)
    env_keys = {
        "TIME_PENALTY": "time_penalty",
        "SCORE_BONUS": "score_bonus",
        "CONCEDE_PENALTY": "concede_penalty",
        "TOUCH_BONUS": "touch_bonus",
    }
    for env_key, param_key in env_keys.items():
        value = os.environ.get(env_key)
        if value is not None:
            params[param_key] = float(value)
    return params


def parse_args():
    parser = argparse.ArgumentParser(
        description="Fine-tune the shaped shared-policy agent against frozen CEIA."
    )
    parser.add_argument(
        "--init-checkpoint",
        default=None,
        help="RLlib checkpoint to initialize the trainable default policy from.",
    )
    parser.add_argument(
        "--ceia-weights",
        default="ceia_baseline_agent/default.pth",
        help="CEIA default.pth path. Copy ceia_baseline_agent into the repo on PACE.",
    )
    parser.add_argument("--ceia-prob", type=float, default=0.85)
    parser.add_argument(
        "--init-shaping",
        default="870",
        choices=["700", "870"],
        help="Which shaped checkpoint family to initialize from when using the default init path.",
    )
    parser.add_argument("--timesteps", type=int, default=15_000_000)
    parser.add_argument("--time-budget-s", type=int, default=28_800)
    parser.add_argument("--checkpoint-freq", type=int, default=25)
    parser.add_argument("--num-workers", type=int, default=8)
    parser.add_argument("--num-envs-per-worker", type=int, default=NUM_ENVS_PER_WORKER)
    parser.add_argument("--name", default="PPO_ceia_finetune")
    parser.add_argument("--local-dir", default="ray_results")
    return parser.parse_args()


def _episode_opponent(episode):
    if episode is None:
        return "ceia" if np.random.random() < CEIA_PROB else "current"
    if "opponent_kind" not in episode.user_data:
        episode.user_data["opponent_kind"] = (
            "ceia" if np.random.random() < CEIA_PROB else "current"
        )
    return episode.user_data["opponent_kind"]


def policy_mapping_fn(agent_id, *args, **kwargs):
    agent_id = int(agent_id)
    if agent_id in (0, 1):
        return "default"

    episode = args[0] if args else kwargs.get("episode")
    opponent_kind = _episode_opponent(episode)
    if opponent_kind == "ceia":
        return "ceia"
    return "default"


def _policy_weights_from_checkpoint(checkpoint_path, policy_id="default"):
    with open(checkpoint_path, "rb") as f:
        trainer_state = pickle.load(f)
    worker_state = pickle.loads(trainer_state["worker"])
    policy_state = worker_state["state"][policy_id]
    return {
        key: value
        for key, value in policy_state.items()
        if isinstance(value, np.ndarray) and key != "_optimizer_variables"
    }


def _policy_weights_from_pth(weights_path):
    raw = torch.load(weights_path, map_location="cpu")
    weights = {}
    for key, value in raw.items():
        if key in ACTION_POLICY_KEYS:
            weights[key] = value.detach().cpu().numpy()
    missing = ACTION_POLICY_KEYS - set(weights)
    if missing:
        raise RuntimeError(f"{weights_path} missing expected policy keys: {missing}")
    return weights


def _write_params(trial_dir, config):
    # build_zip.py only needs multiagent.policies_to_train and model metadata.
    serializable = {
        "multiagent": {
            "policies_to_train": ["default"],
            "policies": {"default": None, "ceia": None},
        },
        "model": config["model"],
    }
    with open(trial_dir / "params.pkl", "wb") as f:
        pickle.dump(serializable, f)
    with open(trial_dir / "params.json", "w") as f:
        json.dump(serializable, f, indent=2)


def _progress_row(result):
    fields = [
        "training_iteration",
        "timesteps_total",
        "episodes_total",
        "episode_reward_mean",
        "episode_reward_min",
        "episode_reward_max",
        "episode_len_mean",
        "time_total_s",
    ]
    return {field: result.get(field) for field in fields}


def main():
    args = parse_args()
    global CEIA_PROB
    CEIA_PROB = float(args.ceia_prob)

    init_checkpoint = Path(args.init_checkpoint) if args.init_checkpoint else None
    if init_checkpoint is None:
        default_root = Path(
            "ray_results/PPO_selfplay_shaped/"
            "PPO_Soccer_d5022_00000_0_2026-04-21_18-24-16"
        )
        init_checkpoint = default_root / (
            "checkpoint_000700/checkpoint-700"
            if args.init_shaping == "700"
            else "checkpoint_000870/checkpoint-870"
        )
    init_checkpoint = Path(init_checkpoint)
    ceia_weights = Path(args.ceia_weights)
    if not init_checkpoint.exists():
        raise SystemExit(f"missing init checkpoint: {init_checkpoint}")
    if not ceia_weights.exists():
        raise SystemExit(
            f"missing CEIA weights: {ceia_weights}. "
            "Copy ceia_baseline_agent/default.pth into the repo or pass --ceia-weights."
        )

    os.environ.setdefault("RAY_DISABLE_DASHBOARD", "1")
    ray.init(include_dashboard=False, ignore_reinit_error=True)
    shaping_overrides = _shaping_overrides_from_env()

    tune.registry.register_env("Soccer", create_rllib_env)
    temp_env = create_rllib_env(
        {
            "shaped_reward": True,
            "shaping_params": shaping_overrides,
            "worker_id": 0,
        }
    )
    obs_space = temp_env.observation_space
    act_space = temp_env.action_space
    temp_env.close()

    config = {
        "num_gpus": 0,
        "num_workers": args.num_workers,
        "num_envs_per_worker": args.num_envs_per_worker,
        "log_level": "INFO",
        "framework": "torch",
        "multiagent": {
            "policies": {
                "default": (None, obs_space, act_space, {}),
                "ceia": (None, obs_space, act_space, {}),
            },
            "policy_mapping_fn": tune.function(policy_mapping_fn),
            "policies_to_train": ["default"],
        },
        "env": "Soccer",
        "env_config": {
            "num_envs_per_worker": args.num_envs_per_worker,
            "shaped_reward": True,
            "shaping_params": shaping_overrides,
        },
        "model": {
            "vf_share_layers": True,
            "fcnet_hiddens": [256, 256],
            "fcnet_activation": "relu",
        },
        "rollout_fragment_length": 1000,
        "batch_mode": "complete_episodes",
    }

    timestamp = time.strftime("%Y-%m-%d_%H-%M-%S")
    trial_dir = Path(args.local_dir) / args.name / f"PPO_Soccer_ceia_00000_0_{timestamp}"
    trial_dir.mkdir(parents=True, exist_ok=True)
    _write_params(trial_dir, config)

    cls = get_trainable_cls("PPO")
    trainer = cls(env="Soccer", config=config)
    ceia_policy_weights = trainer.get_weights(["ceia"])["ceia"]
    ceia_policy_weights.update(_policy_weights_from_pth(str(ceia_weights)))
    trainer.set_weights(
        {
            "default": _policy_weights_from_checkpoint(str(init_checkpoint), "default"),
            "ceia": ceia_policy_weights,
        }
    )

    progress_path = trial_dir / "progress.csv"
    result_path = trial_dir / "result.json"
    with open(progress_path, "w", newline="") as progress_file, open(
        result_path, "w"
    ) as result_file:
        writer = csv.DictWriter(
            progress_file,
            fieldnames=[
                "training_iteration",
                "timesteps_total",
                "episodes_total",
                "episode_reward_mean",
                "episode_reward_min",
                "episode_reward_max",
                "episode_len_mean",
                "time_total_s",
            ],
        )
        writer.writeheader()

        while True:
            result = trainer.train()
            row = _progress_row(result)
            writer.writerow(row)
            progress_file.flush()
            result_file.write(json.dumps(row) + "\n")
            result_file.flush()

            iteration = int(result.get("training_iteration", 0))
            timesteps = int(result.get("timesteps_total", 0))
            elapsed = float(result.get("time_total_s", 0.0))
            reward = result.get("episode_reward_mean")
            print(
                f"iter={iteration} ts={timesteps} time_s={elapsed:.1f} "
                f"reward={reward}"
            )

            if iteration % args.checkpoint_freq == 0:
                print(trainer.save(str(trial_dir)))

            if timesteps >= args.timesteps or elapsed >= args.time_budget_s:
                break

    print(trainer.save(str(trial_dir)))
    print(trial_dir)
    print("Done training")
    trainer.stop()
    ray.shutdown()


if __name__ == "__main__":
    main()
