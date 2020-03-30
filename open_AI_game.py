import gym
import random
import numpy as np 
import tflearn
from tflearn.layers.core import input_data, dropout, fully_connected
from tflearn.layers.estimator import regression
from statistics import mean, median
from collections import Counter

LR = 1e-3 #learning rate
env = gym.make('CartPole-v0') #selecting a game environment
env.reset()
goal_steps = 500
score_requirement = 50
initial_games = 10000

def random_game_first():
    for episode in range(5): # number of games
        env.reset()
        for t in range(goal_steps):
            env.render()
            action = env.action_space.sample()
            observation, reward, done, info = env.step(action)
            if done:
                break

def initial_population():
    training_data = [] #observation and moves made
    scores = []
    accepted_scores = [] #scores crossing the 50 points barrier
    
    # number of games being played
    for _ in range(initial_games):
        score = 0
        game_memory = [] #storing all scores until game end
        prev_observation = []

        # below code is for one entire game
        for _ in range(goal_steps): #max number of steps to get result
            action = random.randrange(0,2) #will generate a 0 or 1 randomly
            observation, reward, done, info = env.step(action)

            if len(prev_observation) > 0:
                game_memory.append([prev_observation, action]) #saving the last observation in game memory

            prev_observation = observation
            score += reward
            if done:
                break

        if score >= score_requirement: # making training data only of acceptable score
            accepted_scores.append(score)
            for data in game_memory: #manually one-hot-encoding the 0,1 output
                if data[1] == 1:
                    output = [0,1]
                elif data[1] == 0:
                    output = [1,0]

                training_data.append([data[0], output])

        env.reset() #reset whenever a game is over
        scores.append(score)

    training_data_save = np.array(training_data)
    np.save('saved.npy', training_data_save) #saving the trainign data in npy arrays

    print('Average accepted score:', mean(accepted_scores))
    print('Median accepted score:', median(accepted_scores))
    print(Counter(accepted_scores)) # seeing amount of data we obtained

    return training_data

initial_population()