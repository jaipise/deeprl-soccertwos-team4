# TEAM4_AGENT_REWARD (Agent2)

**Role:** Reward/observation modification (40-pt rubric item).

## Authors
**Authors:** Jai Pise — jpise3@gatech.edu
**Authors:** Naman Tellakula — ntellakula3@gatech.edu

## Description
PPO self-play trained with a `ShapedRewardWrapper` (see `utils.py`) that adds:
- `+0.01 × (ball_x - prev_ball_x)` — reward ball progress toward opponent goal (x=+14)
- `-0.001` — per-step penalty to discourage stalling
- Original sparse goal reward is preserved.

Hypothesis: the sparse goal reward alone makes credit assignment nearly impossible
in 2v2 soccer; dense progress shaping should accelerate early learning and yield
higher terminal win rate vs. the unshaped baseline (Agent1).

## Training
- Algorithm: PPO (Ray RLlib 1.4.0)
- Script: `train_ray_selfplay_shaped.py`
- Env: `soccer_twos` default (multiagent_player) + `ShapedRewardWrapper`
- Hyperparameters: fcnet_hiddens=[256,256], relu, vf_share_layers=True,
  rollout_fragment_length=5000, complete_episodes batch mode, 15M timesteps / 8h cap
