import ray
from ray import tune
from soccer_twos import EnvType
from soccer_twos.side_channels import EnvConfigurationChannel

from utils import create_rllib_env

env_channel = EnvConfigurationChannel()

NUM_ENVS_PER_WORKER = 3


if __name__ == "__main__":
    ray.init()

    tune.registry.register_env("Soccer", create_rllib_env)
    temp_env = create_rllib_env(
        {"variation": EnvType.multiagent_player, "env_channel": env_channel}
    )
    obs_space = temp_env.observation_space
    act_space = temp_env.action_space
    temp_env.close()

    analysis = tune.run(
        "PPO",
        name="PPO_selfplay_1",
        config={
            "num_gpus": 1,
            "num_workers": 6,
            "log_level": "INFO",
            "framework": "torch",
            "env": "Soccer",
            "input": "/home/bryan/Documents/ceia/course/tournament-starter/data/processed",
            "input_evaluation": [],
            "explore": False,
        },
        stop={
            "timesteps_total": 15000000,
        },
        checkpoint_freq=100,
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
