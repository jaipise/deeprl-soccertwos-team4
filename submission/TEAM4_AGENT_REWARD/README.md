# TEAM4_AGENT_REWARD

**Agent name:** TEAM4_AGENT_REWARD (Agent2)

**Authors**
- Jai Pise  jpise3@gatech.edu
- Naman Tellakula  ntellakula3@gatech.edu

## Rubric mapping
- **Agent2 -- reward / observation modification (40 pts).** Trained from scratch
  with a custom `ShapedRewardWrapper` on top of the standard `soccer_twos`
  environment.

## Description
PPO self-play with a single shared "default" policy and a
rolling 3-generation opponent archive. The reward signal augments the original
sparse goal reward with small dense shaping terms designed to make credit
assignment tractable in 2v2 soccer:

- Ball progress toward the opponent goal (team-aware sign).
- Per-step penalty to discourage stalling.
- Ball-pressure reward for staying close enough to influence the ball.
- Team-spacing reward to encourage passing lanes and reduce clumping.
- Goalie positioning / blocking reward for staying between the ball and own goal.
- Striker support-positioning reward for staying behind the ball in an attacking lane.

Training script: `train_ray_selfplay_shaped.py` (in the repo root).
Wrapper implementation: `utils.ShapedRewardWrapper`.
Hyperparameters: `fcnet_hiddens=[256, 256]`, `fcnet_activation=relu`,
`vf_share_layers=True`, `rollout_fragment_length=5000`,
`batch_mode=complete_episodes`, stop at 15 M timesteps or 8 h wall.

## Files in this folder
- `agent.py` — `TeamAgent(AgentInterface)` with pure-torch inference (no Ray).
- `default.pth` — torch state_dict extracted from the RLlib checkpoint
  (see `../extract_weights.py`). Both teammates share this single policy at
  inference, matching how it was trained.
- `__init__.py` — re-exports `TeamAgent`.