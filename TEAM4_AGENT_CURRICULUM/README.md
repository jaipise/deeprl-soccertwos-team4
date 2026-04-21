# TEAM4_AGENT_CURRICULUM (Agent3)

**Role:** Novel learning concept (+5 rubric item) and primary policy-performance candidate.

## Authors
- Jai Pise - jpise3@gatech.edu
- Naman Tellakula - ntellakula3@gatech.edu

## Description
PPO trained with a multi-agent curriculum over initial game states. Training starts
with close-range finishing scenarios, then advances through midfield attack,
defensive recovery, and full-field randomized play when mean reward crosses each
stage threshold. The learner uses separate striker and goalie policies for the two
controlled teammates, plus a rolling archive of past striker/goalie opponent
snapshots.

## Training
- Algorithm: PPO (Ray RLlib 1.4.0)
- Script: `train_ray_curriculum_multiagent.py`
- Env: `soccer_twos` multi-agent player mode + `ShapedRewardWrapper`
- Curriculum: `curriculum_multiagent.yaml`
- Resources: CPU only, intended for `scripts/soccerstwos_curriculum.batch`
