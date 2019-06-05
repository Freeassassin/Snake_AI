import gym
import gym_chrome_dino
from gym_chrome_dino.utils.wrappers import make_dino

env = gym.make('ChromeDinoNoBrowser-v0')
done = True
while True:
	if done:
		env.reset()
		action = env.action_space.sample()
		observation, reward, done, info = env.step(action)