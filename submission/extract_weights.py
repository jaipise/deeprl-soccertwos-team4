import os
import pickle
import sys

import numpy as np
import torch

import ray.rllib  # noqa: F401


def main():
    ckpt_path, out_dir = sys.argv[1], sys.argv[2]
    pids = sys.argv[3:] or ["striker", "goalie"]
    with open(ckpt_path, "rb") as f:
        trainer_state = pickle.load(f)
    worker_state = pickle.loads(trainer_state["worker"])
    policy_states = worker_state["state"]
    os.makedirs(out_dir, exist_ok=True)
    for pid in pids:
        if pid not in policy_states:
            print(f"[warn] policy '{pid}' missing from checkpoint; skipping")
            continue
        state = policy_states[pid]
        tensor_sd = {
            k: torch.from_numpy(np.asarray(v))
            for k, v in state.items()
            if isinstance(v, np.ndarray) and k != "_optimizer_variables"
        }
        out_path = os.path.join(out_dir, f"{pid}.pth")
        torch.save(tensor_sd, out_path)
        print(f"[ok] {pid}: {len(tensor_sd)} tensors -> {out_path}")
        for k, v in list(tensor_sd.items())[:4]:
            print(f"       {k}: {tuple(v.shape)}")


if __name__ == "__main__":
    main()
