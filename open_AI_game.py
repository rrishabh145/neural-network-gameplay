import gym
import random
import numpy as np 
import tflearn
from tflearn.layers.core import input_data, dropout, fully_connected
from tflearn.layers.estimator import regression
from statistics import mean, median
from collection import Counter

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