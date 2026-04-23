"""Pure-torch inference agent for TEAM4_AGENT_ULTRA_250 (no Ray at inference time).

Shared single policy: both teammates act with the same trained `default` policy,
matching how the trial was trained (policy_mapping_fn maps agent 0 and 1 to
"default").
"""
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
    """Mirrors RLlib FullyConnectedNetwork(fcnet_hiddens=[256,256], activation=relu)."""

    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(OBS_DIM, 256)
        self.fc2 = nn.Linear(256, 256)
        self.logits = nn.Linear(256, sum(ACTION_NVEC))

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        return self.logits(x)


_RLLIB_KEY_MAP = {
    "_hidden_layers.0._model.0.weight": "fc1.weight",
    "_hidden_layers.0._model.0.bias":   "fc1.bias",
    "_hidden_layers.1._model.0.weight": "fc2.weight",
    "_hidden_layers.1._model.0.bias":   "fc2.bias",
    "_logits._model.0.weight":          "logits.weight",
    "_logits._model.0.bias":            "logits.bias",
}


def _load_rllib_weights(model, path):
    raw = torch.load(path, map_location="cpu")
    remapped = {_RLLIB_KEY_MAP[k]: v for k, v in raw.items() if k in _RLLIB_KEY_MAP}
    missing = set(_RLLIB_KEY_MAP.values()) - set(remapped)
    if missing:
        raise RuntimeError(f"Weights file {path} missing keys {missing}")
    model.load_state_dict(remapped)
    model.eval()


class TeamAgent(AgentInterface):
    def __init__(self, env):
        self.name = "TEAM4_AGENT_ULTRA_250"
        self.policy = _PPOPolicy()
        _load_rllib_weights(self.policy, os.path.join(HERE, "default.pth"))

    def act(self, observation):
        actions = {}
        with torch.no_grad():
            for pid, obs in observation.items():
                x = torch.from_numpy(np.asarray(obs, dtype=np.float32)).unsqueeze(0)
                logits = self.policy(x).squeeze(0).numpy()
                parts = np.split(logits, _LOGIT_SPLITS)
                actions[pid] = np.array([int(np.argmax(p)) for p in parts], dtype=np.int64)
        return actions
