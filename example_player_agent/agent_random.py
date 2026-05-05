from typing import Dict

import gym
import numpy as np
from soccer_twos import AgentInterface


class RandomAgent(AgentInterface):
    def __init__(self, env: gym.Env):
        super().__init__()

        self.action_space = env.action_space

    def act(self, observation: Dict[int, np.ndarray]) -> Dict[int, np.ndarray]:
        actions = {}
        for player_id in observation:
            actions[player_id] = self.action_space.sample()
        return actions
