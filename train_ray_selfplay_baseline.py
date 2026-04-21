import ray
from ray import tune
from utils import create_rllib_env


NUM_ENVS_PER_WORKER = 3


if __name__ == "__main__":
    ray.init(include_dashboard=False)

    tune.registry.register_env("Soccer", create_rllib_env)
    temp_env = create_rllib_env()
    obs_space = temp_env.observation_space
    act_space = temp_env.action_space
    temp_env.close()

    tune.run(
        "PPO",
        name="PPO_selfplay_baseline",
        config={
            "num_gpus": 0,
            "num_workers": 8,
            "num_envs_per_worker": NUM_ENVS_PER_WORKER,
            "log_level": "INFO",
            "framework": "torch",
            "multiagent": {
                "policies": {
                    "default": (None, obs_space, act_space, {}),
                },
                "policy_mapping_fn": tune.function(lambda *_: "default"),
                "policies_to_train": ["default"],
            },
            "env": "Soccer",
            "env_config": {"num_envs_per_worker": NUM_ENVS_PER_WORKER},
            "model": {
                "vf_share_layers": True,
                "fcnet_hiddens": [256, 256],
                "fcnet_activation": "relu",
            },
            "rollout_fragment_length": 5000,
            "batch_mode": "complete_episodes",
        },
        stop={"timesteps_total": 15000000, "time_total_s": 28800},
        checkpoint_freq=100,
        checkpoint_at_end=True,
        local_dir="./ray_results",
    )
    print("Done training")
