import os
import sys
import gym
import sneks
import time
from time import sleep
import numpy as np
from numpy import array


goal_steps = 1000
score_requirement = 1
initial_games = 10000
env = gym.make("snek-v1")
env.reset()

training_data = []
scores = []
accepted_scores = []

for _ in range(initial_games):
    #env.reset()
    #snake_head = ()
    food=()
    body=[]
    wall=[]
    score = 0
    game_memory = []
    prev_observation = []
    for i in range(goal_steps):
        #env.render()
        #sleep(1.01)
        action = env.action_space.sample()
        
        observation, reward, done, info = env.step(action)
        
        if len(prev_observation) > 0 :
            game_memory.append([prev_observation, action])
            prev_observation = []
        
       
        for j in np.nditer(observation, flags=['external_loop'], order='F'):
        
            for i in np.nditer(j):

                prev_observation.append(int(i))

        score+=reward
        if done: break


    if score >= score_requirement:
        accepted_scores.append(score)
        output = []
        for data in game_memory:
            if data[1] == 1:
                output = [0,1,0,0]
            elif data[1] == 0:
                output = [1,0,0,0]
            elif data[1] == 2:
                output = [0,0,1,0]
            elif data[1] == 3:
                output = [0,0,0,1]

            training_data.append([data[0], output])
    env.reset()
    scores.append(score)  
File_object = open(r"training_data.txt","a")
for i in training_data:
    File_object.write(str(i))
    File_object.write(",") 
File_object.close() 
"""
[[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 
 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
  0, 0, 255, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 101, 100, 0, 0, 0, 0,
   0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 100, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 100, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 
     n0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 1]]
"""