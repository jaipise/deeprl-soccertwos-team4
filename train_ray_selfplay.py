import numpy as np
import ray
from ray import tune
from ray.rllib.agents.callbacks import DefaultCallbacks
from utils import create_rllib_env


NUM_ENVS_PER_WORKER = 3
OPPONENT_GENERATIONS = ["current", "gen_1", "gen_2", "gen_3"]
OPPONENT_PROBS = [0.50, 0.25, 0.125, 0.125]


def _opponent_generation(episode):
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
    generation = _opponent_generation(episode)
    if generation == "current":
        return role
    return f"opponent_{role}_{generation}"


class SelfPlayUpdateCallback(DefaultCallbacks):
    def on_train_result(self, **info):
        episode_reward_mean = info["result"].get("episode_reward_mean")
        if episode_reward_mean is not None and episode_reward_mean > 0.5:
            print("---- Updating role-specialized opponent archive!!! ----")
            trainer = info["trainer"]
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


if __name__ == "__main__":
    ray.init(include_dashboard=False)

    tune.registry.register_env("Soccer", create_rllib_env)
    temp_env = create_rllib_env()
    obs_space = temp_env.observation_space
    act_space = temp_env.action_space
    temp_env.close()

    analysis = tune.run(
        "PPO",
        name="PPO_selfplay_rec",
        config={
            "num_gpus": 0,
            "num_workers": 8,
            "num_envs_per_worker": NUM_ENVS_PER_WORKER,
            "log_level": "INFO",
            "framework": "torch",
            "callbacks": SelfPlayUpdateCallback,
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
            "env_config": {"num_envs_per_worker": NUM_ENVS_PER_WORKER,},
            "model": {
                "vf_share_layers": True,
                "fcnet_hiddens": [256, 256],
                "fcnet_activation": "relu",
            },
            "rollout_fragment_length": 5000,
            "batch_mode": "complete_episodes",
        },
        stop={"timesteps_total": 15000000, "time_total_s": 28800,},
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
