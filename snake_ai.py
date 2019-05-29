import gym
import sneks
import time
from time import sleep
import random
import numpy as np
from numpy import array
import tensorflow
import tflearn
from tflearn.layers.core import input_data, dropout, fully_connected
from tflearn.layers.estimator import regression
from statistics import median, mean
from collections import Counter

def cal_Distance(x1,y1,x2,y2):
    dist = np.sqrt((x2 - x1)**2 + (y2 - y1)**2)
    return dist

def body_Dis(snake_head,body):
    for i in body:
        return cal_Distance(snake_head[0],snake_head[1],i[0],i[1])

LR = 1e-4  
goal_steps = 500
score_requirement = 1
initial_games = 1000
env = gym.make("snek-v1")
env.reset()

def initial_population():
    # [OBS, MOVES]
    training_data = []
    # all scores:
    scores = []
    accepted_scores = []

    for _ in range(initial_games):
        snake_head = ()
        food=()
        body=[]
        wall=[]
        score = 0
        game_memory = []
        prev_observation = []
        for i in range(goal_steps):

            action = env.action_space.sample()
            
            observation, reward, done, info = env.step(action)
            #env.render()
            #sleep(0.01)
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
            #print(snake_head,body,food)
            body.sort()
            try:
                prev_observation = [cal_Distance(snake_head[0],snake_head[1],food[0],food[1]),
                            	body_Dis(snake_head,body), snake_head[0],snake_head[1],16- snake_head[0], 16 - snake_head[1]]
            except:
                prev_observation = [0,
                                body_Dis(snake_head,body), snake_head[0],snake_head[1],16- snake_head[0], 16 - snake_head[1]]
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
    #print(game_memory,"\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n")
    return training_data

def neural_network_model(input_size):

    network = input_data(shape=[None, input_size, 1], name='input')

    network = fully_connected(network, 128, activation='relu')
    network = dropout(network, 0.8)

    network = fully_connected(network, 256, activation='relu')
    network = dropout(network, 0.8)

    network = fully_connected(network, 512, activation='relu')
    network = dropout(network, 0.8)

    network = fully_connected(network, 256, activation='relu')
    network = dropout(network, 0.8)

    network = fully_connected(network, 128, activation='relu')
    network = dropout(network, 0.8)

    network = fully_connected(network, 4, activation='softmax')
    network = regression(network, optimizer='adam', learning_rate=LR, loss='categorical_crossentropy', name='targets')
    model = tflearn.DNN(network, tensorboard_dir='log')

    return model

def train_model(training_data, model=False):

    X = np.array([i[0] for i in training_data]).reshape(-1,6,1)
    y = [i[1] for i in training_data]

    if not model:
        model = neural_network_model(input_size = len(X[0]))
    
    model.fit({'input': X}, {'targets': y}, n_epoch=3, snapshot_step=500, show_metric=True, run_id='openai_learning')
    return model

training_data = initial_population()
model = train_model(training_data)

scores = []
choices = []
for each_game in range(10):
    snake_head = ()
    food=()
    body=[]
    wall=[]
    score = 0
    game_memory = []
    prev_obs = []
    env.reset()
    for _ in range(goal_steps):
        env.render()
        sleep(0.1)
        if len(prev_obs)==0:
            action = random.randrange(0,2)
        else:
            action = np.argmax(model.predict(prev_obs.reshape(-1,6,1))[0])        

        choices.append(action)
                
        observation, reward, done, info = env.step(action)
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
        #print(snake_head,body,food)
        body.sort()
        try:
            prev_obs = [cal_Distance(snake_head[0],snake_head[1],food[0],food[1]), body_Dis(snake_head,body), snake_head[0],snake_head[1],16- snake_head[0], 16 - snake_head[1]]
        except:
            prev_obs = [0, body_Dis(snake_head,body), snake_head[0],snake_head[1],16- snake_head[0], 16 - snake_head[1]]            

        #game_memory.append([new_observation, action])
        score+=reward
        if done: break

    scores.append(score)

print('Average Score:',sum(scores)/len(scores))
print('choice 1:{}  choice 0:{}'.format(choices.count(1)/len(choices),choices.count(0)/len(choices)))
print(score_requirement)