import gym
import sneks
import time
from time import sleep
import keyboard
done = False
env = gym.make("snek-v1")
env.reset()
action = 0

while not done:
	sleep(0.3)
	if keyboard.is_pressed('w'):
		action = 0
	elif keyboard.is_pressed('d'):
		action = 1
	elif keyboard.is_pressed('s'):
		action = 2
	elif keyboard.is_pressed('a'):
		action = 3
	env.render()
	
	observation, reward, done, info = env.step(action)
print(reward)