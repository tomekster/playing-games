import gym
env = gym.make("MsPacman-v0")
env.reset()

total_reward = 0
for _ in range(10000):
	env.render()
	o, r, done, info = env.step(env.action_space.sample())
	total_reward += r
	if done:
		print("Total reward: {}".format(total_reward))
		total_reward=0
		env.reset()
env.close()
