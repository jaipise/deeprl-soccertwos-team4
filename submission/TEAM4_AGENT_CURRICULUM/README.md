# TEAM4_AGENT_CURRICULUM

**Agent name:** TEAM4_AGENT_CURRICULUM (Agent3)

**Authors**
- Jai Pise  jpise3@gatech.edu
- Naman Tellakula  ntellakula3@gatech.edu

## Rubric mapping
- **Agent3 -- novel concept of learning (+5 pts)** and primary policy-performance candidate.
- Reward/observation modification (40 pts) is carried by the companion `TEAM4_AGENT_REWARD` submission; this agent is *also* trained with that same `ShapedRewardWrapper` so it builds on the reward change.

## Description
PPO (Ray RLlib 1.4.0) trained with a multi-agent curriculum over initial-state distributions and role-specialized policies:

- **Curriculum** — training starts with close-range finishing scenarios, then advances through midfield attack, defensive recovery, and full-field randomized play when the running mean reward crosses each stage's threshold.
- **Role specialization** — the two controlled teammates use *separate* policies ("striker" for agent 0, "goalie" for agent 1) so they can learn distinct behaviors rather than a single shared policy.
- **Self-play archive** — rolling archive of 3 past generations of each role (striker / goalie); opponents sample 50 % from the current learner, 25 % / 12.5 % / 12.5 % from gen_1 / gen_2 / gen_3 snapshots. Archive promotes when mean reward > 0.35 and at least 5 iterations since the last promotion.
- **Reward shaping** — ball-progress shaping (team-aware sign) plus a step penalty, via `utils.ShapedRewardWrapper`.

Training script: `train_ray_curriculum_multiagent.py` (in the repo root).
Hyperparameters: `fcnet_hiddens=[256, 256]`, `fcnet_activation=relu`, `vf_share_layers=True`, `rollout_fragment_length=5000`, `batch_mode=complete_episodes`, stop at 15 M timesteps or 8 h wall.

## Files in this folder
- `agent.py` — `TeamAgent(AgentInterface)` with pure-torch inference (no Ray).
- `striker.pth`, `goalie.pth` — torch state_dicts extracted from the RLlib checkpoint (see `../extract_weights.py`).
- `__init__.py` — re-exports `TeamAgent`.

## Why pure-torch inference
The Gradescope autograder runs ~10 parallel evaluator processes in a Docker container with ~64 MB `/dev/shm`. Calling `ray.init()` per process overruns the Ray object store and kills the Redis handshake (`ImportThread: Connection closed by server`), which made the previous Ray-based agent fail with exit code 1. Stripping Ray from inference avoids the entire issue.
