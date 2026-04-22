import argparse
import os

import numpy as np
import ray
from ray import tune
from ray.rllib.agents.callbacks import DefaultCallbacks
import yaml

from utils import SHAPING_VARIANTS, create_rllib_env


NUM_ENVS_PER_WORKER = 2
OPPONENT_GENERATIONS = ["current", "gen_1", "gen_2", "gen_3"]
OPPONENT_PROBS = [0.50, 0.25, 0.125, 0.125]
ARCHIVE_UPDATE_REWARD = 0.35
ARCHIVE_UPDATE_MIN_ITERATIONS = 5


def load_curriculum():
    with open("curriculum_multiagent.yaml", "r", encoding="utf-8") as f:
        return yaml.load(f, Loader=yaml.FullLoader)["tasks"]


def slurm_cpus(default=16):
    return int(os.environ.get("SLURM_CPUS_PER_TASK", default))


def ray_init_for_ice():
    os.environ.setdefault("RAY_USAGE_STATS_ENABLED", "0")
    os.environ.setdefault("RAY_DISABLE_DASHBOARD", "1")
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    try:
        ray.init(
            include_dashboard=False,
            ignore_reinit_error=True,
            log_to_driver=False,
            num_cpus=slurm_cpus(),
            _node_ip_address="127.0.0.1",
        )
    except TypeError:
        ray.init(
            include_dashboard=False,
            ignore_reinit_error=True,
            log_to_driver=False,
            num_cpus=slurm_cpus(),
        )


def opponent_generation(episode):
    if episode is None:
        return np.random.choice(OPPONENT_GENERATIONS, p=OPPONENT_PROBS)
    if "opponent_generation" not in episode.user_data:
        episode.user_data["opponent_generation"] = np.random.choice(
            OPPONENT_GENERATIONS, p=OPPONENT_PROBS
        )
    return episode.user_data["opponent_generation"]


def policy_mapping_fn(agent_id, *args, **kwargs):
    agent_id = int(agent_id)
    if agent_id == 0:
        return "striker"
    if agent_id == 1:
        return "goalie"

    episode = args[0] if args else kwargs.get("episode")
    role = "striker" if agent_id == 2 else "goalie"
    generation = opponent_generation(episode)
    if generation == "current":
        return role
    return f"opponent_{role}_{generation}"


def set_curriculum_stage(trainer, stage):
    def set_worker_stage(worker):
        worker.foreach_env(lambda env: env.set_curriculum_task(stage))

    trainer.workers.local_worker().foreach_env(lambda env: env.set_curriculum_task(stage))
    trainer.workers.foreach_worker(set_worker_stage)


class CurriculumSelfPlayCallback(DefaultCallbacks):
    def on_train_result(self, **info):
        result = info["result"]
        episode_reward_mean = result.get("episode_reward_mean")
        if episode_reward_mean is None:
            return

        trainer = info["trainer"]
        tasks = trainer.config["env_config"]["curriculum_tasks"]
        stage = getattr(trainer, "_curriculum_stage", 0)

        if stage < len(tasks) - 1 and episode_reward_mean >= tasks[stage]["threshold"]:
            stage += 1
            trainer._curriculum_stage = stage
            set_curriculum_stage(trainer, stage)
            print(f"---- Curriculum advanced to {stage}: {tasks[stage]['name']} ----")

        iteration = result.get("training_iteration", 0)
        last_update = getattr(trainer, "_last_archive_update_iteration", -999)
        if (
            episode_reward_mean >= ARCHIVE_UPDATE_REWARD
            and iteration - last_update >= ARCHIVE_UPDATE_MIN_ITERATIONS
        ):
            trainer._last_archive_update_iteration = iteration
            print("---- Updating role-specialized opponent archive!!! ----")
            trainer.set_weights(
                {
                    "opponent_striker_gen_3": trainer.get_weights(
                        ["opponent_striker_gen_2"]
                    )["opponent_striker_gen_2"],
                    "opponent_goalie_gen_3": trainer.get_weights(
                        ["opponent_goalie_gen_2"]
                    )["opponent_goalie_gen_2"],
                    "opponent_striker_gen_2": trainer.get_weights(
                        ["opponent_striker_gen_1"]
                    )["opponent_striker_gen_1"],
                    "opponent_goalie_gen_2": trainer.get_weights(
                        ["opponent_goalie_gen_1"]
                    )["opponent_goalie_gen_1"],
                    "opponent_striker_gen_1": trainer.get_weights(["striker"])[
                        "striker"
                    ],
                    "opponent_goalie_gen_1": trainer.get_weights(["goalie"])[
                        "goalie"
                    ],
                }
            )


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--variant",
        default=os.environ.get("SHAPING_VARIANT", "V0_baseline"),
        choices=sorted(SHAPING_VARIANTS),
        help="Reward shaping variant to train. Also settable via $SHAPING_VARIANT.",
    )
    parser.add_argument(
        "--timesteps",
        type=int,
        default=int(os.environ.get("TIMESTEPS_TOTAL", 15000000)),
    )
    parser.add_argument(
        "--time-budget-s",
        type=int,
        default=int(os.environ.get("TIME_BUDGET_S", 28800)),
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    ray_init_for_ice()

    tasks = load_curriculum()
    tune.registry.register_env("Soccer", create_rllib_env)
    temp_env = create_rllib_env({"curriculum_tasks": tasks})
    obs_space = temp_env.observation_space
    act_space = temp_env.action_space
    temp_env.close()

    worker_count = min(8, max(1, slurm_cpus() // 2))
    print(f"---- Training variant: {args.variant} ----")
    analysis = tune.run(
        "PPO",
        name=f"PPO_curriculum_multiagent_{args.variant}",
        config={
            "num_gpus": 0,
            "num_workers": worker_count,
            "num_envs_per_worker": NUM_ENVS_PER_WORKER,
            "log_level": "INFO",
            "framework": "torch",
            "callbacks": CurriculumSelfPlayCallback,
            "multiagent": {
                "policies": {
                    "striker": (None, obs_space, act_space, {}),
                    "goalie": (None, obs_space, act_space, {}),
                    "opponent_striker_gen_1": (None, obs_space, act_space, {}),
                    "opponent_goalie_gen_1": (None, obs_space, act_space, {}),
                    "opponent_striker_gen_2": (None, obs_space, act_space, {}),
                    "opponent_goalie_gen_2": (None, obs_space, act_space, {}),
                    "opponent_striker_gen_3": (None, obs_space, act_space, {}),
                    "opponent_goalie_gen_3": (None, obs_space, act_space, {}),
                },
                "policy_mapping_fn": tune.function(policy_mapping_fn),
                "policies_to_train": ["striker", "goalie"],
            },
            "env": "Soccer",
            "env_config": {
                "num_envs_per_worker": NUM_ENVS_PER_WORKER,
                "curriculum_tasks": tasks,
                "curriculum_task_index": 0,
                "shaped_reward": True,
                "shaping_variant": args.variant,
            },
            "model": {
                "vf_share_layers": True,
                "fcnet_hiddens": [256, 256],
                "fcnet_activation": "relu",
            },
            "rollout_fragment_length": 5000,
            "batch_mode": "complete_episodes",
        },
        stop={"timesteps_total": args.timesteps, "time_total_s": args.time_budget_s},
        checkpoint_freq=25,
        checkpoint_at_end=True,
        local_dir="./ray_results",
    )

    best_trial = analysis.get_best_trial("episode_reward_mean", mode="max")
    print(best_trial)
    if best_trial is not None:
        best_checkpoint = analysis.get_best_checkpoint(
            trial=best_trial, metric="episode_reward_mean", mode="max"
        )
        print(best_checkpoint)
    print("Done training")
