"""Build a submission zip from an RLlib trial directory.

Auto-detects:
  - Latest checkpoint in the trial dir (highest checkpoint_NNNNNN).
  - Trainable policy names from params.pkl (multiagent.policies_to_train).
  - Architecture: must be FullyConnectedNetwork(fcnet_hiddens=[256,256], relu).
    Anything else (LSTM / different hidden sizes) aborts with a clear message.

Template selection by trainable-policy set:
  {'default'}             -> single shared policy (both teammates share)
  {'striker', 'goalie'}   -> role-specialized (agent 0 = striker, agent 1 = goalie)

Usage:
    python submission/build_zip.py \
        --trial ray_results/PPO_curriculum_multiagent/PPO_Soccer_6efa0_00000_0_2026-04-21_17-24-09 \
        --name TEAM4_AGENT_CURRICULUM \
        --rubric "Agent3 -- novel concept of learning (+5 pts)" \
        --desc "PPO with curriculum of initial-state distributions + role-specialized striker/goalie policies + self-play archive + reward shaping."

Produces:
    submission/<NAME>/__init__.py
    submission/<NAME>/agent.py
    submission/<NAME>/README.md
    submission/<NAME>/<policy>.pth   (one per trainable policy)
    submission/<NAME>.zip
"""
import argparse
import glob
import os
import pickle
import shutil
import subprocess
import sys
import textwrap

import numpy as np
import torch

import ray.rllib  # noqa: F401  (pickle needs filter classes importable)


HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(HERE)

REQUIRED_RLLIB_KEYS = {
    "_hidden_layers.0._model.0.weight",
    "_hidden_layers.0._model.0.bias",
    "_hidden_layers.1._model.0.weight",
    "_hidden_layers.1._model.0.bias",
    "_logits._model.0.weight",
    "_logits._model.0.bias",
}

AUTHORS_BLOCK = (
    "- Jai Pise  jpise3@gatech.edu\n"
    "- Naman Tellakula  ntellakula3@gatech.edu\n"
)


AGENT_SHARED_TEMPLATE = '''"""Pure-torch inference agent for {name} (no Ray at inference time).

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


_RLLIB_KEY_MAP = {{
    "_hidden_layers.0._model.0.weight": "fc1.weight",
    "_hidden_layers.0._model.0.bias":   "fc1.bias",
    "_hidden_layers.1._model.0.weight": "fc2.weight",
    "_hidden_layers.1._model.0.bias":   "fc2.bias",
    "_logits._model.0.weight":          "logits.weight",
    "_logits._model.0.bias":            "logits.bias",
}}


def _load_rllib_weights(model, path):
    raw = torch.load(path, map_location="cpu")
    remapped = {{_RLLIB_KEY_MAP[k]: v for k, v in raw.items() if k in _RLLIB_KEY_MAP}}
    missing = set(_RLLIB_KEY_MAP.values()) - set(remapped)
    if missing:
        raise RuntimeError(f"Weights file {{path}} missing keys {{missing}}")
    model.load_state_dict(remapped)
    model.eval()


class TeamAgent(AgentInterface):
    def __init__(self, env):
        self.name = "{name}"
        self.policy = _PPOPolicy()
        _load_rllib_weights(self.policy, os.path.join(HERE, "default.pth"))

    def act(self, observation):
        actions = {{}}
        with torch.no_grad():
            for pid, obs in observation.items():
                x = torch.from_numpy(np.asarray(obs, dtype=np.float32)).unsqueeze(0)
                logits = self.policy(x).squeeze(0).numpy()
                parts = np.split(logits, _LOGIT_SPLITS)
                actions[pid] = np.array([int(np.argmax(p)) for p in parts], dtype=np.int64)
        return actions
'''


AGENT_ROLES_TEMPLATE = '''"""Pure-torch inference agent for {name} (no Ray at inference time).

Role-specialized policies: the lower-id teammate acts with the `striker` policy,
the higher-id teammate with the `goalie` policy, matching how the trial was
trained.
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


_RLLIB_KEY_MAP = {{
    "_hidden_layers.0._model.0.weight": "fc1.weight",
    "_hidden_layers.0._model.0.bias":   "fc1.bias",
    "_hidden_layers.1._model.0.weight": "fc2.weight",
    "_hidden_layers.1._model.0.bias":   "fc2.bias",
    "_logits._model.0.weight":          "logits.weight",
    "_logits._model.0.bias":            "logits.bias",
}}


def _load_rllib_weights(model, path):
    raw = torch.load(path, map_location="cpu")
    remapped = {{_RLLIB_KEY_MAP[k]: v for k, v in raw.items() if k in _RLLIB_KEY_MAP}}
    missing = set(_RLLIB_KEY_MAP.values()) - set(remapped)
    if missing:
        raise RuntimeError(f"Weights file {{path}} missing keys {{missing}}")
    model.load_state_dict(remapped)
    model.eval()


class TeamAgent(AgentInterface):
    def __init__(self, env):
        self.name = "{name}"
        self.striker = _PPOPolicy()
        self.goalie = _PPOPolicy()
        _load_rllib_weights(self.striker, os.path.join(HERE, "striker.pth"))
        _load_rllib_weights(self.goalie, os.path.join(HERE, "goalie.pth"))

    def act(self, observation):
        order = sorted(observation)
        actions = {{}}
        with torch.no_grad():
            for pid, obs in observation.items():
                model = self.striker if pid == order[0] else self.goalie
                x = torch.from_numpy(np.asarray(obs, dtype=np.float32)).unsqueeze(0)
                logits = model(x).squeeze(0).numpy()
                parts = np.split(logits, _LOGIT_SPLITS)
                actions[pid] = np.array([int(np.argmax(p)) for p in parts], dtype=np.int64)
        return actions
'''


def pick_latest_checkpoint(trial_dir):
    ckpt_dirs = sorted(glob.glob(os.path.join(trial_dir, "checkpoint_*")))
    if not ckpt_dirs:
        raise SystemExit(f"no checkpoints in {trial_dir}")
    latest = ckpt_dirs[-1]
    inner = glob.glob(os.path.join(latest, "checkpoint-*"))
    inner = [p for p in inner if not p.endswith(".tune_metadata")]
    if not inner:
        raise SystemExit(f"no checkpoint-N file inside {latest}")
    return inner[0]


def read_trainable_policies(trial_dir):
    params = os.path.join(trial_dir, "params.pkl")
    with open(params, "rb") as f:
        cfg = pickle.load(f)
    ma = cfg.get("multiagent", {}) or {}
    ptt = ma.get("policies_to_train") or list((ma.get("policies") or {}).keys())
    model = cfg.get("model", {}) or {}
    return list(ptt), model


def extract_weights(ckpt_path, out_dir, policy_ids):
    with open(ckpt_path, "rb") as f:
        trainer_state = pickle.load(f)
    worker_state = pickle.loads(trainer_state["worker"])
    policy_states = worker_state["state"]
    os.makedirs(out_dir, exist_ok=True)
    wrote = []
    for pid in policy_ids:
        if pid not in policy_states:
            print(f"[warn] policy '{pid}' missing from checkpoint; skipping")
            continue
        state = policy_states[pid]
        tensor_sd = {
            k: torch.from_numpy(np.asarray(v))
            for k, v in state.items()
            if isinstance(v, np.ndarray) and k != "_optimizer_variables"
        }
        present = set(tensor_sd.keys())
        missing = REQUIRED_RLLIB_KEYS - present
        if missing:
            raise SystemExit(
                f"policy '{pid}' missing expected FFN keys {missing}; "
                "this builder only supports fcnet_hiddens=[256,256]. "
                "If this trial uses LSTM/other architecture, it needs a custom agent.py."
            )
        out_path = os.path.join(out_dir, f"{pid}.pth")
        torch.save(tensor_sd, out_path)
        print(f"[ok] {pid}: {len(tensor_sd)} tensors -> {out_path}")
        wrote.append(pid)
    return wrote


def choose_template(trainable_pids, name):
    tset = set(trainable_pids)
    if tset == {"default"}:
        return AGENT_SHARED_TEMPLATE.format(name=name), ["default"]
    if tset == {"striker", "goalie"}:
        return AGENT_ROLES_TEMPLATE.format(name=name), ["striker", "goalie"]
    raise SystemExit(
        f"unsupported trainable-policy set {tset} for {name}. "
        "Supported: {'default'} or {'striker','goalie'}. "
        "Edit build_zip.py to add a new template if needed."
    )


def write_readme(out_dir, name, rubric, desc, trial_dir, ckpt_path,
                 trainable_pids, extra_notes=""):
    rel_trial = os.path.relpath(trial_dir, REPO)
    rel_ckpt = os.path.relpath(ckpt_path, REPO)
    pth_lines = "\n".join(f"- `{pid}.pth` — state_dict for the `{pid}` policy." for pid in trainable_pids)
    body = textwrap.dedent(f"""\
        # {name}

        **Agent name:** {name}

        **Authors**
        {AUTHORS_BLOCK}
        ## Rubric mapping
        - {rubric}

        ## Description
        {desc}

        Source trial: `{rel_trial}`
        Checkpoint packaged: `{rel_ckpt}`

        Hyperparameters: `fcnet_hiddens=[256, 256]`, `fcnet_activation=relu`,
        `vf_share_layers=True`, `rollout_fragment_length=5000`,
        `batch_mode=complete_episodes`.

        ## Files in this folder
        - `agent.py` — `TeamAgent(AgentInterface)` with pure-torch inference (no Ray).
        {pth_lines}
        - `__init__.py` — re-exports `TeamAgent`.
        {extra_notes}
        """)
    with open(os.path.join(out_dir, "README.md"), "w") as f:
        f.write(body)


def write_init(out_dir):
    with open(os.path.join(out_dir, "__init__.py"), "w") as f:
        f.write("from .agent import TeamAgent\n")


def write_agent_py(out_dir, agent_src):
    with open(os.path.join(out_dir, "agent.py"), "w") as f:
        f.write(agent_src)


def zip_dir(submission_root, name):
    zip_path = os.path.join(submission_root, f"{name}.zip")
    if os.path.exists(zip_path):
        os.remove(zip_path)
    # zip from inside submission_root so paths are "<name>/..." not "submission/<name>/..."
    subprocess.check_call(
        ["zip", "-r", f"{name}.zip", name,
         "-x", "*.DS_Store", "*/__pycache__/*"],
        cwd=submission_root,
    )
    return zip_path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--trial", required=True, help="path to PPO_Soccer_<id>_... trial dir")
    ap.add_argument("--name", required=True, help="submission name, e.g. TEAM4_AGENT_CURRICULUM")
    ap.add_argument("--rubric", default="", help="one-line rubric mapping for README")
    ap.add_argument("--desc", default="", help="free-form description for README")
    ap.add_argument("--ckpt", default=None, help="override checkpoint path (else picks latest)")
    ap.add_argument("--extra-notes", default="", help="appended to README after why-pure-torch")
    args = ap.parse_args()

    trial_dir = os.path.abspath(args.trial)
    if not os.path.isdir(trial_dir):
        sys.exit(f"not a dir: {trial_dir}")

    ckpt_path = args.ckpt or pick_latest_checkpoint(trial_dir)
    print(f"[info] trial  = {trial_dir}")
    print(f"[info] ckpt   = {ckpt_path}")

    trainable_pids, model_cfg = read_trainable_policies(trial_dir)
    hiddens = model_cfg.get("fcnet_hiddens")
    if hiddens and list(hiddens) != [256, 256]:
        sys.exit(f"fcnet_hiddens={hiddens} != [256,256]; update build_zip.py to support.")
    if model_cfg.get("use_lstm"):
        sys.exit("trial uses LSTM; needs a custom agent.py — not handled by build_zip.py.")
    print(f"[info] trainable policies = {trainable_pids}")

    agent_src, use_pids = choose_template(trainable_pids, args.name)

    out_dir = os.path.join(HERE, args.name)
    if os.path.exists(out_dir):
        shutil.rmtree(out_dir)
    os.makedirs(out_dir)

    wrote = extract_weights(ckpt_path, out_dir, use_pids)
    missing = set(use_pids) - set(wrote)
    if missing:
        sys.exit(f"failed to extract required policies {missing}")

    write_agent_py(out_dir, agent_src)
    write_init(out_dir)
    write_readme(
        out_dir,
        name=args.name,
        rubric=args.rubric or f"{args.name} submission.",
        desc=args.desc or "See training script for details.",
        trial_dir=trial_dir,
        ckpt_path=ckpt_path,
        trainable_pids=use_pids,
        extra_notes=args.extra_notes,
    )
    zip_path = zip_dir(HERE, args.name)
    print(f"[done] wrote {zip_path}")


if __name__ == "__main__":
    main()
