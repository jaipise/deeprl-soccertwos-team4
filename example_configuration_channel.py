import random

import soccer_twos
from soccer_twos import EnvType


env = soccer_twos.make(
    base_port=8500,
    watch=True,
    flatten_branched=True,
    variation=EnvType.team_vs_policy,
    single_player=True,
    opponent_policy=lambda *_: 0,
)
print("Observation Space: ", env.observation_space.shape)
print("Action Space: ", env.action_space)

step = 0
team0_reward = 0
team1_reward = 0
env.reset()

env.env_channel.set_parameters(
    players_states={
        1: {"rotation_y": 45, "position": [-14, 1.5],},
    },
)


while True:
    obs, reward, done, info = env.step(26)

    if step == 30:
        print("updating policy")
        env.set_opponent_policy(lambda *_: env.action_space.sample())

    step += 1
    if done:
        step = 0
        team0_reward = 0
        team1_reward = 0
        env.reset()
