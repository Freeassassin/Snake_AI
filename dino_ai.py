import gym
import sneks
import time
from time import sleep
import random
import numpy as np
from numpy import array



"""
import tflearn
from tflearn.layers.core import input_data, dropout, fully_connected
from tflearn.layers.estimator import regression
from statistics import median, mean
from collections import Counter
"""
def cal_Distance(x1,y1,x2,y2):
    dist = np.sqrt((x2 - x1)**2 + (y2 - y1)**2)
    return dist
LR = 1e-4  
goal_steps = 10
score_requirement = 0
initial_games = 1
env = gym.make("snek-v1")
env.reset()

def initial_population():
    # [OBS, MOVES]
    training_data = []
    # all scores:
    scores = []
    accepted_scores = []
    snake_head = ()
    food=()
    body=[]
    wall=[]
    for _ in range(initial_games):
        observation = env.reset()
        score = 0
        game_memory = []
        game_observation = []
        prev_observation = []
        for i in range(goal_steps):
            action = env.action_space.sample()
            observation, reward, done, info = env.step(action)
            #env.render()
            #sleep(0.1) 
            """        
            if len(prev_observation) > 0 :
                game_memory.append([prev_observation, action])
"""
            game_observation = observation
            prev_observation = []
            snake_head, food = (), ()
            body, wall = [], []
            y= 0
            x= 0
            for j in np.nditer(game_observation, flags=['external_loop'], order='F'):
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
            print(snake_head,body,food)
            prev_observation.append([cal_Distance(snake_head[0],snake_head[1],food[0],food[1]),
            	cal_Distance(snake_head[0],snake_head[1],body[0][0],body[0][1]),
            	cal_Distance(snake_head[0],snake_head[1],body[1][0],body[1][1]),
            	cal_Distance(snake_head[0],snake_head[1],body[2][0],body[2][1]),
            	snake_head[0],snake_head[1],16- snake_head[0], 16 - snake_head[1]])
            print(prev_observation)
            score+=reward
            if done: break


        env.reset()

"""
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

        scores.append(score)  
    # some stats here, to further illustrate the neural network magic!
    #print('Average accepted score:',mean(accepted_scores))
    #print('Median score for accepted scores:',median(accepted_scores))
    #print(Counter(accepted_scores))
    #print(game_memory,"\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n")
    #print(training_data,"\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n")
"""
training_data = initial_population()

"""
[
    [
        array(
               [
                    [  0.,   0.,   0.,   0.,   0.,   0.,   0.,   0.,   0.,   0.,   0.,  0.,   0.,   0.,   0.,   0.],
                    [  0.,   0.,   0.,   0.,   0.,   0.,   0.,   0.,   0.,   0.,   0.,  0.,   0.,   0.,   0.,   0.],
                    [  0.,   0.,   0.,   0.,   0.,   0.,   0.,   0.,   0.,   0.,   0.,  0.,   0.,   0.,   0.,   0.],
                    [  0.,   0.,   0.,   0.,   0.,   0.,   0.,   0.,   0.,   0.,   0.,  0.,   0.,   0.,   0.,   0.],
                    [  0.,   0.,   0.,   0.,   0.,   0.,   0.,   0., 100., 101.,   0.,  0.,   0.,   0.,   0.,   0.],
                    [  0.,   0.,   0.,   0.,   0.,   0.,   0.,   0., 100.,   0.,   0.,  0.,   0.,   0.,   0.,   0.],
                    [  0.,   0.,   0.,   0.,   0.,   0.,   0.,   0., 100.,   0.,   0.,  0., 255.,   0.,   0.,   0.],
                    [  0.,   0.,   0.,   0.,   0.,   0.,   0.,   0.,   0.,   0.,   0.,  0.,   0.,   0.,   0.,   0.],
                    [  0.,   0.,   0.,   0.,   0.,   0.,   0.,   0.,   0.,   0.,   0.,  0.,   0.,   0.,   0.,   0.],
                    [  0.,   0.,   0.,   0.,   0.,   0.,   0.,   0.,   0.,   0.,   0.,  0.,   0.,   0.,   0.,   0.],
                    [  0.,   0.,   0.,   0.,   0.,   0.,   0.,   0.,   0.,   0.,   0.,  0.,   0.,   0.,   0.,   0.],
                    [  0.,   0.,   0.,   0.,   0.,   0.,   0.,   0.,   0.,   0.,   0.,  0.,   0.,   0.,   0.,   0.],
                    [  0.,   0.,   0.,   0.,   0.,   0.,   0.,   0.,   0.,   0.,   0.,  0.,   0.,   0.,   0.,   0.],
                    [  0.,   0.,   0.,   0.,   0.,   0.,   0.,   0.,   0.,   0.,   0.,  0.,   0.,   0.,   0.,   0.],
                    [  0.,   0.,   0.,   0.,   0.,   0.,   0.,   0.,   0.,   0.,   0.,  0.,   0.,   0.,   0.,   0.],
                    [  0.,   0.,   0.,   0.,   0.,   0.,   0.,   0.,   0.,   0.,   0.,  0.,   0.,   0.,   0.,   0.]]



                    )]
                    """