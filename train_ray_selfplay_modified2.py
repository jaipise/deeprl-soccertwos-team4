import numpy as np
import ray
from ray import tune
from ray.rllib.agents.callbacks import DefaultCallbacks
from utils import create_rllib_env, sample_player
import yaml


NUM_ENVS_PER_WORKER = 3


def policy_mapping_fn(agent_id, *args, **kwargs):
    if int(agent_id) in (0, 1):
        return "default"
    return np.random.choice(
        ["default", "opponent_1", "opponent_2", "opponent_3"],
        p=[0.50, 0.25, 0.125, 0.125],
    )


class CurriculumSelfPlayCallback(DefaultCallbacks):
    def __init__(self):
        super().__init__()
        with open("curriculum.yaml", "r") as f:
            self.curriculum = yaml.safe_load(f)
        self.current = 0

    def on_episode_start(self, worker, base_env, policies, episode, **kwargs):
        env = base_env.get_sub_environments()[0]
        task = self.curriculum[self.current]

        env.reset_config = {
            "ball": sample_player(task["ball"]),
            "team0": [sample_player(task["player"]), sample_player(task["player"])],
            "team1": [sample_player(task["player"]), sample_player(task["player"])],
        }

    def on_train_result(self, trainer, result, **kwargs):
        reward = result["episode_reward_mean"]

        # self-play update
        if reward > 0.5:
            trainer.set_weights({
                "opponent_3": trainer.get_weights(["opponent_2"])["opponent_2"],
                "opponent_2": trainer.get_weights(["opponent_1"])["opponent_1"],
                "opponent_1": trainer.get_weights(["default"])["default"],
            })

        # curriculum progression
        if reward > 1.5 and self.current < len(self.curriculum) - 1:
            self.current += 1
            print(f"Advancing curriculum → level {self.current}")


if __name__ == "__main__":
    ray.init()

    tune.registry.register_env("Soccer", create_rllib_env)

    temp_env = create_rllib_env({"shaped_reward": True})
    obs_space = temp_env.observation_space
    act_space = temp_env.action_space
    temp_env.close()

    tune.run(
        "PPO",
        name="PPO_selfplay_shaped_curriculum",
        config={
            "env": "Soccer",
            "env_config": {
                "shaped_reward": True,
                "num_envs_per_worker": NUM_ENVS_PER_WORKER,
            },

            "num_workers": 8,
            "num_envs_per_worker": NUM_ENVS_PER_WORKER,
            "framework": "torch",

            "callbacks": CurriculumSelfPlayCallback,

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

            # strong PPO settings
            "lr": 5e-5,
            "gamma": 0.99,
            "lambda": 0.95,
            "clip_param": 0.2,
            "num_sgd_iter": 10,
            "sgd_minibatch_size": 4096,
            "train_batch_size": 65536,

            "model": {
                "fcnet_hiddens": [256, 256],
                "vf_share_layers": True,
            },
        },
        stop={"training_iteration": 500},
        checkpoint_freq=25,
        local_dir="ray_results",
    )