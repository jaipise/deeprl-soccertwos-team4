import os
import numpy as np
import torch
import torch.nn as nn

from soccer_twos import AgentInterface


HERE = os.path.dirname(os.path.abspath(__file__))
OBS_DIM = 336
ACTION_NVEC = (3, 3, 3)
_LOGIT_SPLITS = np.cumsum(ACTION_NVEC)[:-1]


class _PPOPolicy(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(OBS_DIM, 256)
        self.fc2 = nn.Linear(256, 256)
        self.logits = nn.Linear(256, sum(ACTION_NVEC))

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        return self.logits(x)


def _load_weights(model, path):
    state = torch.load(path, map_location="cpu")
    model.load_state_dict(state)
    model.eval()


class TeamAgent(AgentInterface):
    def __init__(self, env):
        self.name = "TEAM4_AGENT_SP_MOD_TORCH"
        self.policy = _PPOPolicy()
        _load_weights(self.policy, os.path.join(HERE, "policy.pth"))

    def act(self, observation):
        actions = {}
        with torch.no_grad():
            for pid, obs in observation.items():
                x = torch.from_numpy(np.asarray(obs, dtype=np.float32)).unsqueeze(0)
                logits = self.policy(x).squeeze(0).numpy()
                parts = np.split(logits, _LOGIT_SPLITS)
                actions[pid] = np.array([int(np.argmax(p)) for p in parts], dtype=np.int64)
        return actions
