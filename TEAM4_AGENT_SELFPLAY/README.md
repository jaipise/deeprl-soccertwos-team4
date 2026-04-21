# TEAM4_AGENT_SELFPLAY (Agent3)

**Role:** Novel concept of learning (+5-pt rubric bonus) and primary policy-performance submission.

## Authors
**Authors:** Jai Pise — jpise3@gatech.edu
**Authors:** Naman Tellakula — ntellakula3@gatech.edu

## Description
PPO trained with **self-play** against a rolling archive of 3 past policy snapshots.
The learner's policy is promoted into the opponent archive whenever mean episode
reward exceeds 0.5, producing progressively harder opponents over the course of
training. Opponent sampling at rollout time: 50% current, 25% gen-1, 12.5% gen-2,
12.5% gen-3.

## Training
- Algorithm: PPO (Ray RLlib 1.4.0) with `SelfPlayUpdateCallback`
- Script: `train_ray_selfplay.py`
- Env: `soccer_twos` default (multiagent_player), unmodified reward
- Hyperparameters: fcnet_hiddens=[256,256], relu, vf_share_layers=True,
  rollout_fragment_length=5000, complete_episodes batch mode, 15M timesteps / 8h cap
