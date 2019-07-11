import os
import sys
import gym
import sneks
import time
import numpy as np
from time import sleep
from numpy import array
from keras import layers
from collections import Counter
from statistics import median, mean
from keras.layers import Dense, Input
from keras.models import Sequential , load_model

"""
training_data = []
training_dataFile = open("training_data.txt", "r")
training_dataFile = training_dataFile.read()
exec(training_dataFile)

x_train = np.array([])
y_train = np.array([])
x = 0

for i in training_data:
    x += 1
    x_train = np.append(x_train, i[0])
    y_train = np.append(y_train, i[1])

x_train = x_train.reshape(x,20)
y_train = y_train.reshape(x,4)
"""
"""
model = Sequential([

    Dense(18, activation = 'relu', input_dim = 20),
    Dense(18, activation = 'relu'),
    Dense(4, activation = 'softmax'),
])
"""
model = load_model('snakeAI-v1.h5')

model.summary()
model.compile(optimizer='adam',loss='categorical_crossentropy',metrics= ['accuracy'])

#model.fit(x_train,y_train,epochs = 200)
#model.save("snakeAI-v3.h5")

def whereFood(x1,y1,x2,y2):
    if x1 == x2:
        if y1 > y2:
            prev_observation.extend([0,0,0,0,y1 - y2,0,0,0])
        elif y2 > y1:
            prev_observation.extend([y2 - y1,0,0,0,0,0,0,0])
        else:
            prev_observation.extend([0,0,0,0,0,0,0,0])
    elif y1 == y2:
        if x1 > x2:
            prev_observation.extend([0,0,0,0,0,0,x1 - x2,0])
        elif x2 > x1:
            prev_observation.extend([0,0,x2 - x1,0,0,0,0,0])
        else:
            prev_observation.extend([0,0,0,0,0,0,0,0])
    elif x1 > x2:
        if (x1 - x2) == (y1 - y2):
            prev_observation.extend([0,0,0,0,0,cal_Distance(x1,y1,x2,y2),0,0])
        elif (x1 - x2) == (y2 - y1):
            prev_observation.extend([0,0,0,0,0,0,0,cal_Distance(x1,y1,x2,y2)])
        else:
            prev_observation.extend([0,0,0,0,0,0,0,0])
    elif x2 > x1:
        if (x2 - x1) == (y2 - y1):
            prev_observation.extend([0,cal_Distance(x1,y1,x2,y2),0,0,0,0,0,0])
        elif (x2 - x1) == (y1 - y2):
            prev_observation.extend([0,0,0,cal_Distance(x1,y1,x2,y2),0,0,0,0])
        else:
            prev_observation.extend([0,0,0,0,0,0,0,0])
    else:
        prev_observation.extend([0,0,0,0,0,0,0,0])

def whereWall(x,y):

    prev_observation.extend([16 - y, x , y, 16 - x])

def whereBod(x,y,body):
    d1 = 0
    d2 = 0
    d3 = 0
    d4 = 0
    d5 = 0
    d6 = 0
    d7 = 0
    d8 = 0
    for i in body:
        if x == i[0]:
            if y > i[1]:
                d5 = y - i[1]
            else:
                d1 = i[1] - y
        if y == i[1]:
            if x > i[0]:
                d7 = x - i[0]
            else:
                d3 = i[0] - x
        if x > i[0]:
            if (x - i[0]) == (y - i[1]):
                d6 = cal_Distance(x,y,i[0],i[1])
            elif (x - i[0]) == (i[1] - y):
                d8 = cal_Distance(x,y,i[0],i[1])
        if i[0] > x:
            if (i[0] - x) == (i[1] - y):
                d2 = cal_Distance(x,y,i[0],i[1])
            elif (i[0] - x) == (y - i[1]):
                d4 = cal_Distance(x,y,i[0],i[1])
    prev_observation.extend([d1,d2,d3,d4,d5,d6,d7,d8])
       
def cal_Distance(x1,y1,x2,y2):
    dist = np.sqrt((x2 - x1)**2 + (y2 - y1)**2)
    return dist


goal_steps = 100
score_requirement = -1
initial_games = 1000
env = gym.make("snek-v1")
env.reset()
training_data = []
scores = []
accepted_scores = []

for _ in range(initial_games):
    #env.reset()
    score = 0
    game_memory = []
    prev_observation = []
    for i in range(goal_steps):
        #env.render()
        #sleep(0.1)
        if not len(prev_observation) > 0 :

            action = env.action_space.sample()
        else:
            x_test = np.array([])

            for i in prev_observation:
                x_test = np.append(x_test, i)

            x_test = x_test.reshape(1,20)

            prediction = model.predict([x_test])
            
            action =np.argmax(prediction[0])

        observation, reward, done, info = env.step(action)

        if len(prev_observation) > 0 :
            game_memory.append([prev_observation, action])
            prev_observation = []
        
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
        
        if food == () or food == None:
            food = snake_head[0],snake_head[1]
        
        body.sort()
        whereFood(snake_head[0],snake_head[1],food[0],food[1])    
        whereBod(snake_head[0],snake_head[1],body)
        whereWall(snake_head[0],snake_head[1])
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

print('Average accepted score:',mean(accepted_scores))
print('Median score for accepted scores:',median(accepted_scores))
print(Counter(accepted_scores))
"""
File_object = open(r"smart_training_data.txt","a")
for i in training_data:
    File_object.write(str(i))
    File_object.write(",") 
File_object.close() 
"""