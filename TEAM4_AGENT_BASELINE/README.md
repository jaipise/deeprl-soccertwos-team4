# TEAM4_AGENT_BASELINE (Agent1)

**Role:** Plain policy baseline (unmodified reward/observation, no archived opponents).

## Authors
**Authors:** Jai Pise — jpise3@gatech.edu
**Authors:** Naman Tellakula — ntellakula3@gatech.edu

## Description
PPO trained as a shared policy for all SoccerTwos agents against the default sparse
reward (goals only). This baseline deliberately does not use shaped rewards,
curriculum learning, or a rolling opponent archive, so it serves as the reference
curve for Agent2 (reward-modded) and Agent3 (archive self-play).

## Training
- Algorithm: PPO (Ray RLlib 1.4.0)
- Script: `train_ray_selfplay_baseline.py`
- Env: `soccer_twos` default (multiagent_player), no shaping, one shared policy
- Hyperparameters: fcnet_hiddens=[256,256], relu, vf_share_layers=True,
  rollout_fragment_length=5000, complete_episodes batch mode, 15M timesteps / 8h cap
