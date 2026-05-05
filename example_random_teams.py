import soccer_twos

env = soccer_twos.make(
    render=True,
    flatten_branched=True,
    variation=soccer_twos.EnvType.team_vs_policy,
    single_player=True,
    opponent_policy=lambda *_: 0,
)
print("Observation Space: ", env.observation_space.shape)
print("Action Space: ", env.action_space)

team0_reward = 0
env.reset()
while True:
    obs, reward, done, info = env.step(env.action_space.sample())
    team0_reward += reward
    if done:
        print("Total Reward: ", team0_reward)
        team0_reward = 0
        env.reset()
