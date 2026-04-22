import os
import numpy as np
import yaml
import ray
from ray import tune
from ray.rllib.agents.callbacks import DefaultCallbacks
from utils import create_rllib_env, sample_pos_vel, sample_player


# Keep this conservative for cluster stability.
NUM_ENVS_PER_WORKER = 1
NUM_WORKERS = 4

CURRICULUM_PATH = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    "curriculum.yaml",
)

config_fns = {
    "none": lambda *_: None,
    "random_players": lambda env: env.set_policies(
        lambda *_: env.action_space.sample()
    ),
}


def policy_mapping_fn(agent_id, *args, **kwargs):
    if int(agent_id) in (0, 1):
        return "default"
    return np.random.choice(
        ["default", "opponent_1", "opponent_2", "opponent_3"],
        p=[0.50, 0.25, 0.125, 0.125],
    )


class SuperiorSelfPlayCallback(DefaultCallbacks):
    def __init__(self):
        super().__init__()
        with open(CURRICULUM_PATH, "r") as f:
            curriculum = yaml.load(f, Loader=yaml.FullLoader)
        self.tasks = curriculum["tasks"]
        self.current = 0
        self.best_reward = float("-inf")
        self.last_archive_update_reward = float("-inf")

    def on_episode_start(
        self, *, worker, base_env, policies, episode, env_index, **kwargs
    ):
        task = self.tasks[self.current]

        for env in base_env.get_unwrapped():
            config_fns[task["config_fn"]](env)
            env.env_channel.set_parameters(
                ball_state=sample_pos_vel(task["ranges"]["ball"]),
                players_states={
                    player: sample_player(task["ranges"]["players"][player])
                    for player in task["ranges"]["players"]
                },
            )

    def on_train_result(self, *, trainer, result, **kwargs):
        reward = result.get("episode_reward_mean", 0.0)

        if reward > self.best_reward:
            self.best_reward = reward

        # Self-play archive update:
        # update only when reward is meaningfully better than last archive refresh.
        if reward > 0.5 and reward > self.last_archive_update_reward + 0.10:
            print("---- Updating opponents!!! ----")
            trainer.set_weights(
                {
                    "opponent_3": trainer.get_weights(["opponent_2"])["opponent_2"],
                    "opponent_2": trainer.get_weights(["opponent_1"])["opponent_1"],
                    "opponent_1": trainer.get_weights(["default"])["default"],
                }
            )
            self.last_archive_update_reward = reward

        # Curriculum progression:
        # advance only after the policy is clearly stronger, not too early.
        if reward > 1.2 and self.current < len(self.tasks) - 1:
            self.current += 1
            print(
                f"---- Advancing curriculum to task {self.current}: "
                f"{self.tasks[self.current]['name']} ----"
            )


if __name__ == "__main__":
    ray.init(include_dashboard=False)

    tune.registry.register_env("Soccer", create_rllib_env)

    temp_env = create_rllib_env(
        {
            "shaped_reward": True,
            "num_envs_per_worker": NUM_ENVS_PER_WORKER,
        }
    )
    obs_space = temp_env.observation_space
    act_space = temp_env.action_space
    temp_env.close()

    analysis = tune.run(
        "PPO",
        name="PPO_selfplay_ensemble",
        config={
            # system
            "num_gpus": 0,
            "num_workers": NUM_WORKERS,
            "num_envs_per_worker": NUM_ENVS_PER_WORKER,
            "framework": "torch",
            "log_level": "INFO",
            "callbacks": SuperiorSelfPlayCallback,

            # environment
            "env": "Soccer",
            "env_config": {
                "shaped_reward": True,
                "num_envs_per_worker": NUM_ENVS_PER_WORKER,
            },

            # self-play
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

            # PPO: slightly more conservative than your collapsing run
            "lr": 3e-5,
            "gamma": 0.99,
            "lambda": 0.95,
            "clip_param": 0.2,
            "entropy_coeff": 0.003,
            "vf_loss_coeff": 1.0,
            "num_sgd_iter": 5,
            "sgd_minibatch_size": 4096,
            "train_batch_size": 32768,
            "rollout_fragment_length": 2000,
            "batch_mode": "complete_episodes",

            # model
            "model": {
                "vf_share_layers": True,
                "fcnet_hiddens": [256, 256],
                "fcnet_activation": "relu",
            },
        },
        stop={
            "training_iteration": 600,
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