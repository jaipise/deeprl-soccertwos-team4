import numpy as np
import ray
from ray import tune
from ray.rllib.agents.callbacks import DefaultCallbacks
from utils import create_rllib_env


NUM_ENVS_PER_WORKER = 3


def policy_mapping_fn(agent_id, *args, **kwargs):
    if int(agent_id) in (0, 1):
        return "default"
    return np.random.choice(
        ["default", "opponent_1", "opponent_2", "opponent_3"],
        p=[0.50, 0.25, 0.125, 0.125],
    )


class SelfPlayUpdateCallback(DefaultCallbacks):
    def on_train_result(self, **info):
        result = info["result"]
        reward = result.get("episode_reward_mean")

        if reward is not None and reward > 0.5:
            print("---- Updating opponents!!! ----")
            trainer = info["trainer"]
            trainer.set_weights(
                {
                    "opponent_3": trainer.get_weights(["opponent_2"])["opponent_2"],
                    "opponent_2": trainer.get_weights(["opponent_1"])["opponent_1"],
                    "opponent_1": trainer.get_weights(["default"])["default"],
                }
            )


if __name__ == "__main__":
    ray.init(include_dashboard=False)

    tune.registry.register_env("Soccer", create_rllib_env)
    temp_env = create_rllib_env({"shaped_reward": True})
    obs_space = temp_env.observation_space
    act_space = temp_env.action_space
    temp_env.close()

    analysis = tune.run(
        "PPO",
        name="PPO_selfplay_shaped_v2",
        config={
            # system
            "num_gpus": 0,
            "num_workers": 8,
            "num_envs_per_worker": NUM_ENVS_PER_WORKER,
            "num_cpus_per_worker": 1,
            "framework": "torch",
            "log_level": "INFO",
            "callbacks": SelfPlayUpdateCallback,

            # environment
            "env": "Soccer",
            "env_config": {
                "num_envs_per_worker": NUM_ENVS_PER_WORKER,
                "shaped_reward": True,
            },

            # multi-agent self-play
            "multiagent": {
                "policies": {
                    "default": (None, obs_space, act_space, {}),
                    "opponent_1": (None, obs_space, act_space, {}),
                    "opponent_2": (None, obs_space, act_space, {}),
                    "opponent_3": (None, obs_space, act_space, {}),
                },
                "policy_mapping_fn": tune.function(policy_mapping_fn),
                "policies_to_train": ["default"],
            },

            # PPO hyperparameters
            "lr": 5e-5,
            "gamma": 0.99,
            "lambda": 0.95,
            "clip_param": 0.2,
            "entropy_coeff": 0.005,
            "vf_loss_coeff": 1.0,
            "num_sgd_iter": 10,
            "sgd_minibatch_size": 4096,
            "train_batch_size": 65536,
            "rollout_fragment_length": 5000,
            "batch_mode": "complete_episodes",

            # model
            "model": {
                "vf_share_layers": True,
                "fcnet_hiddens": [256, 256],
                "fcnet_activation": "relu",
            },
        },
        stop={
            "training_iteration": 1000,
            "time_total_s": 28800,
        },
        checkpoint_freq=25,
        checkpoint_at_end=True,
        keep_checkpoints_num=5,
        checkpoint_score_attr="episode_reward_mean",
        local_dir="./ray_results",
    )

    best_trial = analysis.get_best_trial("episode_reward_mean", mode="max")
    print(best_trial)
    best_checkpoint = analysis.get_best_checkpoint(
        trial=best_trial, metric="episode_reward_mean", mode="max"
    )
    print(best_checkpoint)
    print("Done training")