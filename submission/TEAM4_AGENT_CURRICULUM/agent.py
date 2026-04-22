import os

import numpy as np
import torch
import torch.nn as nn

from soccer_twos import AgentInterface


HERE = os.path.dirname(os.path.abspath(__file__))
OBS_DIM = 336
ACTION_NVEC = (3, 3, 3)  # MultiDiscrete branches
_LOGIT_SPLITS = np.cumsum(ACTION_NVEC)[:-1]


class _PPOPolicy(nn.Module):
    """Mirrors RLlib FullyConnectedNetwork(fcnet_hiddens=[256,256], activation=relu).

    RLlib param names use nested SlimFC wrappers; this module uses the same names
    (fc1/fc2/logits) and a key remap is applied in load_rllib_weights below.
    """

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
    """Agent3 -- curriculum-trained striker/goalie PPO team (pure-torch inference)."""

    def __init__(self, env):
        self.name = "TEAM4_AGENT_CURRICULUM"
        self.striker = _PPOPolicy()
        self.goalie = _PPOPolicy()
        _load_rllib_weights(self.striker, os.path.join(HERE, "striker.pth"))
        _load_rllib_weights(self.goalie, os.path.join(HERE, "goalie.pth"))

    def act(self, observation):
        order = sorted(observation)
        actions = {}
        with torch.no_grad():
            for pid, obs in observation.items():
                model = self.striker if pid == order[0] else self.goalie
                x = torch.from_numpy(np.asarray(obs, dtype=np.float32)).unsqueeze(0)
                logits = model(x).squeeze(0).numpy()
                parts = np.split(logits, _LOGIT_SPLITS)
                actions[pid] = np.array([int(np.argmax(p)) for p in parts], dtype=np.int64)
        return actions
