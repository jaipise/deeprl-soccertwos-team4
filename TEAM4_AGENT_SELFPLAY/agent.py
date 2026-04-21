import os

import ray
from ray.rllib.agents.ppo import PPOTrainer
from soccer_twos import AgentInterface


CHECKPOINT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "checkpoint")


class TeamAgent(AgentInterface):
    """Agent3 — PPO with 4-snapshot self-play opponent archive (novel concept)."""

    def __init__(self, env):
        self.name = "TEAM4_AGENT_SELFPLAY"
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
                    "default": (None, env.observation_space, env.action_space, {}),
                },
                "policy_mapping_fn": lambda *_: "default",
            },
        })
        if os.path.exists(CHECKPOINT):
            self.trainer.restore(CHECKPOINT)
        else:
            print(f"Checkpoint not found at {CHECKPOINT}; using untrained policy.")

    def act(self, observation):
        return {
            pid: self.trainer.compute_single_action(
                obs, policy_id="default", explore=False
            )
            for pid, obs in observation.items()
        }
