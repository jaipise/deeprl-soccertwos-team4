# TEAM4_AGENT_SELFPLAY (Ablation)

**Role:** Optional ablation for comparing role-specialized self-play against the curriculum agent.

## Authors
**Authors:** Jai Pise — jpise3@gatech.edu
**Authors:** Naman Tellakula — ntellakula3@gatech.edu

## Description
PPO trained with **role-specialized league self-play**:
- Separate trainable `striker` and `goalie` policies are learned instead of one shared
  policy for both teammates.
- Opponents are sampled from a rolling archive of 3 frozen striker/goalie policy pairs.
- The current striker/goalie pair is promoted into the archive whenever mean episode
  reward exceeds 0.5, producing progressively harder and more diverse opponents.

Opponent sampling at rollout time: 50% current, 25% gen-1, 12.5% gen-2, 12.5% gen-3.

## Training
- Algorithm: PPO (Ray RLlib 1.4.0) with role-specialized policies and `SelfPlayUpdateCallback`
- Script: `train_ray_selfplay.py`
- Env: `soccer_twos` default (multiagent_player), unmodified reward
- Hyperparameters: fcnet_hiddens=[256,256], relu, vf_share_layers=True,
  rollout_fragment_length=5000, complete_episodes batch mode, 15M timesteps / 8h cap
