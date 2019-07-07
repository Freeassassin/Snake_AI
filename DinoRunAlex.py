import gym
import gym_chrome_dino
from gym_chrome_dino.utils.wrappers import make_dino
import numpy as np
from numpy import array
from collections import Counter
from statistics import median, mean
env = gym.make('ChromeDinoNoBrowser-v0')
env.reset()
env = make_dino(env, timer=True, frame_stack=True)
done = False


InitGames = 1
training_data = []
scoreReq = 0
scores = []
accepted_scores = []



for i in range(InitGames):

    score = 0
    game_memory = []
    prev_observation = []
    env.reset()
    done = False 
    while not done:
        #env.render()
        #sleep(0.1)

        action = env.action_space.sample()
        observation, reward,done, info = env.step(action)        

        if len(prev_observation) > 0 :
            game_memory.append([prev_observation, action])
            prev_observation = []
        
        """
        for j in np.nditer(observation, flags=['external_loop'], order='F'):
        
            for i in np.nditer(j):

                prev_observation.append(int(i))
        """
        prev_observation 
        score = env.unwrapped.game.get_score()
        if done: break


    if score >= scoreReq:
        accepted_scores.append(score)
        output = []
        for data in game_memory:
            if data[1] == 1:
                output = [0,1,0,0]
            elif data[1] == 0:
                output = [1,0,0,0]
            elif data[1] == 2:
                output = [0,0,1,0]


            training_data.append([data[0], output])
    env.reset()
    scores.append(score)  

print('Average accepted score:',mean(accepted_scores))
print('Median score for accepted scores:',median(accepted_scores))
print(Counter(accepted_scores))

File_object = open(r"dino_train.txt","a")
for i in training_data:
    File_object.write(str(i))
    File_object.write(",") 
File_object.close() 
