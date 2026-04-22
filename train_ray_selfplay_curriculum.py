import numpy as np
import yaml
import ray
from ray import tune
from ray.rllib.agents.callbacks import DefaultCallbacks
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


def policy_mapping_fn(agent_id, *args, **kwargs):
    if agent_id == 0:
        return "default"  # Choose 01 policy for agent_01
    else:
        return np.random.choice(
            ["default", "opponent_1", "opponent_2", "opponent_3"],
            size=1,
            p=[0.50, 0.25, 0.125, 0.125],
        )[0]


class SelfPlayCurriculumCallback(DefaultCallbacks):
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
        """
        Update multiagent opponent weights and curriculum task when reward is high enough
        """
        global current
        result = info["result"]
        reward = result["episode_reward_mean"]
        if reward > 0.5:
            print("---- Updating opponents!!! ----")
            trainer = info["trainer"]
            trainer.set_weights(
                {
                    "opponent_3": trainer.get_weights(["opponent_2"])["opponent_2"],
                    "opponent_2": trainer.get_weights(["opponent_1"])["opponent_1"],
                    "opponent_1": trainer.get_weights(["default"])["default"],
                }
            )
        if reward > 1.5:
            if current < len(tasks) - 1:
                print("---- Updating curriculum task!!! ----")
                current += 1
                print(f"Current task: {current} - {tasks[current]['name']}")


if __name__ == "__main__":
    ray.init(include_dashboard=False)

    tune.registry.register_env("Soccer", create_rllib_env)
    temp_env = create_rllib_env()
    obs_space = temp_env.observation_space
    act_space = temp_env.action_space
    temp_env.close()

    analysis = tune.run(
        "PPO",
        name="PPO_selfplay_curriculum",
        config={
            # system settings
            "num_gpus": 0,
            "num_workers": 8,
            "num_envs_per_worker": NUM_ENVS_PER_WORKER,
            "log_level": "INFO",
            "framework": "torch",
            "callbacks": SelfPlayCurriculumCallback,
            # RL setup
            "multiagent": {
                "policies": {
                    "default": (None, obs_space, act_space, {}),
                    "opponent_1": (None, obs_space, act_space, {}),
                    "opponent_2": (None, obs_space, act_space, {}),
                    "opponent_3": (None, obs_space, act_space, {}),
                },
                "policy_mapping_fn": policy_mapping_fn,
            },
            # PPO settings
            "lr": 5e-5,
            "gamma": 0.99,
            "lambda": 0.95,
            "clip_param": 0.2,
            "num_sgd_iter": 10,
            "sgd_minibatch_size": 4096,
            "train_batch_size": 65536,
            "model": {
                "vf_share_layers": True,
                "fcnet_hiddens": [256, 256],
                "fcnet_activation": "relu",
            },
            "env": "Soccer",
        },
        stop={"training_iteration": 1000},
        checkpoint_freq=50,
        checkpoint_at_end=True,
    )