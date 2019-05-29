import os
import sys
import gym
import sneks
import time
from time import sleep
import random
import numpy as np
from numpy import array
import tflearn
from tflearn.layers.core import input_data, dropout, fully_connected
from tflearn.layers.estimator import regression
from statistics import median, mean
from collections import Counter


LR = 1e-4  
goal_steps = 1000
score_requirement = 1
initial_games = 10000
env = gym.make("snek-v1")

def cal_Distance(x1,y1,x2,y2):
    dist = np.sqrt((x2 - x1)**2 + (y2 - y1)**2)
    return dist

training_data = []
scores = []
accepted_scores = []

for _ in range(initial_games):
    env.reset()
    snake_head = ()
    food=()
    body=[]
    wall=[]
    score = 0
    game_memory = []
    prev_observation = []
    for i in range(goal_steps):
        env.render()
        sleep(0.01)
        action = env.action_space.sample()
        
        observation, reward, done, info = env.step(action)

        if len(prev_observation) > 0 :
            game_memory.append([prev_observation, action])
        
        snake_head = ()
        food = ()
        
        body= []
        wall = [] 
        
        y= 0
        x= 0
        
        for j in np.nditer(observation, flags=['external_loop'], order='F'):
        
            x+=1
            y = 0
        
            for i in np.nditer(j):
        
                y += 1
        
                if i == 101.:
                    snake_head = x,y
        
                elif i == 255.:
                    food = x,y
        
                elif i == 100.:
                    body.append((x,y))
        body.sort()
        try:
            prev_observation = [cal_Distance(snake_head[0],snake_head[1],food[0],food[1]), snake_head[0],snake_head[1],16- snake_head[0], 16 - snake_head[1]]
            for i in body:
                prev_observation.append(cal_Distance(snake_head[0],snake_head[1],i[0],i[1]))
        except:
            prev_observation = [0, snake_head[0],snake_head[1],16- snake_head[0], 16 - snake_head[1]]
            for i in body:
                prev_observation.append(cal_Distance(snake_head[0],snake_head[1],i[0],i[1]))
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
# some stats here, to further illustrate the neural network magic!
print('Average accepted score:',mean(accepted_scores))
print('Median score for accepted scores:',median(accepted_scores))
print(Counter(accepted_scores))

File_object = open(r"training_data.txt","a")
for i in training_data:
    File_object.write(str(i))
    File_object.write(",") 
File_object.close() 