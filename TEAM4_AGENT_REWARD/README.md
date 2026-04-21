# TEAM4_AGENT_REWARD (Agent2)

**Role:** Reward/observation modification (40-pt rubric item).

## Authors
**Authors:** Jai Pise — jpise3@gatech.edu
**Authors:** Naman Tellakula — ntellakula3@gatech.edu

## Description
PPO self-play trained with a `ShapedRewardWrapper` (see `utils.py`) that preserves
the original sparse goal reward and adds small dense shaping terms:
- Ball progress toward the opponent goal.
- A per-step penalty to discourage stalling.
- Ball-pressure reward for staying close enough to influence the ball.
- Team-spacing reward to encourage passing lanes and reduce clumping.
- Goalie positioning and blocking rewards for staying between the ball and own goal.
- Striker support-positioning reward for staying behind the ball in an attacking lane.

Hypothesis: the sparse goal reward alone makes credit assignment nearly impossible
in 2v2 soccer; dense tactical shaping should accelerate early learning by rewarding
intermediate soccer behaviors that correlate with scoring and defending.

## Training
- Algorithm: PPO (Ray RLlib 1.4.0)
- Script: `train_ray_selfplay_shaped.py`
- Env: `soccer_twos` default (multiagent_player) + `ShapedRewardWrapper`
- Hyperparameters: fcnet_hiddens=[256,256], relu, vf_share_layers=True,
  rollout_fragment_length=5000, complete_episodes batch mode, 15M timesteps / 8h cap
