# TEAM4_AGENT_BASELINE (Agent1)

**Role:** Policy-performance baseline (unmodified reward/observation).

## Authors
**Authors:** Jai Pise — jpise3@gatech.edu
**Authors:** Naman Tellakula — ntellakula3@gatech.edu

## Description
PPO trained with self-play (4-snapshot opponent archive) against the default sparse
reward (goals only). Serves as the reference curve in the comparison plot against
Agent2 (reward-modded) and Agent3 (self-play primary).

## Training
- Algorithm: PPO (Ray RLlib 1.4.0)
- Script: `train_ray_selfplay_baseline.py`
- Env: `soccer_twos` default (multiagent_player), no shaping
- Hyperparameters: fcnet_hiddens=[256,256], relu, vf_share_layers=True,
  rollout_fragment_length=5000, complete_episodes batch mode, 15M timesteps / 8h cap
