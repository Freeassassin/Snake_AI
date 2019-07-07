import os
import sys
import gym
import sneks
import time
import numpy as np
from time import sleep
from numpy import array
#import tensorflow as tf
from keras import layers
from collections import Counter
from statistics import median, mean
from keras.layers import Dense, Input
from keras.models import Sequential , load_model

"""
training_data= []
training_dataFile = open("smart_training_data.txt", "r")
training_dataFile = training_dataFile.read()
exec(training_dataFile)

x_train = np.array([])
y_train = np.array([])
x = 0

for i in training_data:
    x += 1
    x_train = np.append(x_train, i[0])
    y_train = np.append(y_train, i[1])

x_train = x_train.reshape(x,256)
y_train = y_train.reshape(x,4)
"""

"""
model = Sequential([

    Dense(50, activation = 'relu', input_dim = 256),
    Dense(50, activation = 'relu'),
    Dense(50,activation= 'relu'),
    Dense(4, activation = 'softmax'),
])
"""
model = load_model('256,40,40,40,4.h5')
model.summary()
model.compile(optimizer='adam',loss='categorical_crossentropy',metrics= ['accuracy'])
#model.fit(x_train,y_train,epochs = 3)

#model.save("3_50.h5")
#model = load_model('my_model.h5')

goal_steps = 100
score_requirement = 1
initial_games = 10000
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
        env.render()
        sleep(0.1)
        if not len(prev_observation) > 0 :

            action = env.action_space.sample()
        else:
            x_test = np.array([])

            for i in prev_observation:
                x_test = np.append(x_test, i)

            x_test = x_test.reshape(1,256)

            prediction = model.predict([x_test])
            
            action =np.argmax(prediction[0])

        observation, reward, done, info = env.step(action)

        if len(prev_observation) > 0 :
            game_memory.append([prev_observation, action])
            prev_observation = []
        

        for j in np.nditer(observation, flags=['external_loop'], order='F'):
            #print(j)
        
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
# some stats here, to further illustrate the neural network magic!
print('Average accepted score:',mean(accepted_scores))
print('Median score for accepted scores:',median(accepted_scores))
print(Counter(accepted_scores))
#training_dataFile.close()
File_object = open(r"smart_training_data.txt","a")
for i in training_data:
    File_object.write(str(i))
    File_object.write(",") 
File_object.close() 