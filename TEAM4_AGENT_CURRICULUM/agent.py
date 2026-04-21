import os

import ray
from ray.rllib.agents.ppo import PPOTrainer
from soccer_twos import AgentInterface


CHECKPOINT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "checkpoint")


OPPONENT_POLICIES = {
    "opponent_striker_gen_1",
    "opponent_goalie_gen_1",
    "opponent_striker_gen_2",
    "opponent_goalie_gen_2",
    "opponent_striker_gen_3",
    "opponent_goalie_gen_3",
}


class TeamAgent(AgentInterface):
    """Agent3 - curriculum-trained striker/goalie PPO team."""

    def __init__(self, env):
        self.name = "TEAM4_AGENT_CURRICULUM"
        if not ray.is_initialized():
            ray.init(
                local_mode=True,
                num_cpus=1,
                include_dashboard=False,
                ignore_reinit_error=True,
                log_to_driver=False,
            )
        self.trainer = PPOTrainer(config={
            "env": None,
            "framework": "torch",
            "num_workers": 0,
            "num_gpus": 0,
            "model": {
                "vf_share_layers": True,
                "fcnet_hiddens": [256, 256],
                "fcnet_activation": "relu",
            },
            "multiagent": {
                "policies": {
                    "striker": (None, env.observation_space, env.action_space, {}),
                    "goalie": (None, env.observation_space, env.action_space, {}),
                    **{
                        policy_id: (
                            None,
                            env.observation_space,
                            env.action_space,
                            {},
                        )
                        for policy_id in OPPONENT_POLICIES
                    },
                },
                "policy_mapping_fn": lambda *_: "striker",
            },
        })
        if os.path.exists(CHECKPOINT):
            self.trainer.restore(CHECKPOINT)
        else:
            print(f"Checkpoint not found at {CHECKPOINT}; using untrained policy.")

    def act(self, observation):
        ordered_players = sorted(observation)
        return {
            pid: self.trainer.compute_single_action(
                obs,
                policy_id="striker" if pid == ordered_players[0] else "goalie",
                explore=False,
            )
            for pid, obs in observation.items()
        }
