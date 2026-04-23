# TEAM4_AGENT_SP_MOD_TORCH

Pure-torch inference agent for the shaped self-play PPO model (SP_MOD).

This submission avoids Ray/RLlib at inference time because the Gradescope
autograder runs many parallel evaluator processes, which can break Ray-based
inference. The policy weights were extracted offline from the RLlib checkpoint
and saved as `policy.pth`.
