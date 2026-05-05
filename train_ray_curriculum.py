import yaml

import ray
from ray import tune
from ray.rllib.agents.callbacks import DefaultCallbacks
from soccer_twos import EnvType

from utils import create_rllib_env, sample_pos_vel, sample_player


NUM_ENVS_PER_WORKER = 3

current = 0
with open("curriculum.yaml") as f:
    curriculum = yaml.load(f, Loader=yaml.FullLoader)
tasks = curriculum["tasks"]
config_fns = {
    "none": lambda *_: None,
    "random_players": lambda env: env.set_policies(
        lambda *_: env.action_space.sample()
    ),
}


class CurriculumUpdateCallback(DefaultCallbacks):
    def on_episode_start(
        self, *, worker, base_env, policies, episode, env_index, **kwargs
    ) -> None:
        global current, tasks

        for env in base_env.get_unwrapped():
            config_fns[tasks[current]["config_fn"]](env)
            env.env_channel.set_parameters(
                ball_state=sample_pos_vel(tasks[current]["ranges"]["ball"]),
                players_states={
                    player: sample_player(tasks[current]["ranges"]["players"][player])
                    for player in tasks[current]["ranges"]["players"]
                },
            )

    def on_train_result(self, **info):
        global current
        if info["result"]["episode_reward_mean"] > 1.5:
            if current < len(tasks) - 1:
                print("---- Updating tasks!!! ----")
                current += 1
                print(f"Current task: {current} - {tasks[current]['name']}")


if __name__ == "__main__":
    ray.init()

    tune.registry.register_env("Soccer", create_rllib_env)
    temp_env = create_rllib_env()
    obs_space = temp_env.observation_space
    act_space = temp_env.action_space
    temp_env.close()

    analysis = tune.run(
        "PPO",
        name="PPO_curriculum",
        config={
            "num_gpus": 1,
            "num_workers": 14,
            "num_envs_per_worker": NUM_ENVS_PER_WORKER,
            "log_level": "INFO",
            "framework": "torch",
            "callbacks": CurriculumUpdateCallback,
            "env": "Soccer",
            "env_config": {
                "num_envs_per_worker": NUM_ENVS_PER_WORKER,
                "variation": EnvType.team_vs_policy,
                "multiagent": False,
                "flatten_branched": True,
                "single_player": True,
                "opponent_policy": lambda *_: 0,
            },
            "model": {
                "vf_share_layers": True,
                "fcnet_hiddens": [256, 256],
                "fcnet_activation": "relu",
            },
            "rollout_fragment_length": 5000,
            "batch_mode": "complete_episodes",
        },
        stop={
            "timesteps_total": 15000000,
            "time_total_s": 7200,
            "episode_reward_mean": 1.9,
        },
        checkpoint_freq=5,
        checkpoint_at_end=True,
        local_dir="./ray_results",
    )

    best_trial = analysis.get_best_trial("episode_reward_mean", mode="max")
    print(best_trial)
    best_checkpoint = analysis.get_best_checkpoint(
        trial=best_trial, metric="episode_reward_mean", mode="max"
    )
    print(best_checkpoint)
    print("Done training")
