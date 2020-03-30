import gym
import random
import numpy as np 
import tflearn
from tflearn.layers.core import input_data, dropout, fully_connected
from tflearn.layers.estimator import regression
from statistics import mean, median
from collections import Counter
import time

LR = 1e-3 #learning rate for NN
env = gym.make('CartPole-v0') #selecting a game environment
env.reset()

goal_steps = 500
score_requirement = 50 #minimum score for a random model to acieve to be included in training data
initial_games = 10000 #number of games worth of data to collect for training

# function to run a game with random values
def random_game_first():
    for episode in range(5): # number of games
        env.reset()
        for t in range(goal_steps):
            env.render()
            action = env.action_space.sample()
            observation, reward, done, info = env.step(action)
            if done:
                break

# function to create initial training data
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
        # include code from this part if retraining of the models needs to be done
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

    print('Average accepted score:', mean(accepted_scores)) #checking the average score of training samples
    print('Median accepted score:', median(accepted_scores))
    print(Counter(accepted_scores)) # seeing amount of data we obtained

    return training_data

# function to define a tf model
def neural_network_model(input_size):
    network = input_data(shape = [None, input_size, 1], name = 'input' ) #initializing the size of input data

    # creating a layered network of NN

    network = fully_connected(network, 128, activation = 'relu')
    network = dropout(network, 0.8) # 0.2 is dropout

    network = fully_connected(network, 256, activation = 'relu')
    network = dropout(network, 0.8)

    network = fully_connected(network, 512, activation = 'relu')
    network = dropout(network, 0.8)

    network = fully_connected(network, 256, activation = 'relu')
    network = dropout(network, 0.8)

    network = fully_connected(network, 128, activation = 'relu')
    network = dropout(network, 0.8)

    network = fully_connected(network, 2, activation = 'softmax') #output is one hot encoded binary
    # initializing optimization variables for the NN
    network = regression(network, optimizer='adam', learning_rate = LR, loss = 'categorical_crossentropy', name = 'targets')

    model = tflearn.DNN(network, tensorboard_dir = 'log')

    return model

# fn to train the tf model using training data declared in previous fn
def train_model(training_data, model = False):
    X = np.array([i[0] for i in training_data]).reshape(-1, len(training_data[0][0]), 1)
    y = [i[1] for i in training_data]

    if not model: #if we are not loading an earlier pretrained model
        model = neural_network_model(input_size = len(X[0])) #getting the raw train_model  

    # fitting the model with the data
    model.fit({'input':X}, {'targets':y},n_epoch = 3, snapshot_step = 500, show_metric = True, run_id = 'OpenAIgame')

    return model

training_data = initial_population()
model = train_model(training_data)

scores = []
choices = []

for each_game in range(100):
    score = 0
    game_memory = []
    prev_obs = []
    env.reset()
    for _ in range(goal_steps):
        # env.render()
        # time.sleep(0.005)
        if len(prev_obs) == 0:
            action = random.randrange(0,2)
        else:
            # print(model.predict(prev_obs.reshape(-1, len(prev_obs),1)))
            action = np.argmax(model.predict(prev_obs.reshape(-1, len(prev_obs),1))[0])
        choices.append(action)

        new_observation, reward, done, info = env.step(action)
        prev_obs = new_observation
        game_memory.append([new_observation,action])
        score += reward
        if done:
            break
    scores.append(score)

print('Average Score', sum(scores)/len(scores))
print('Choice 1: {}, Choice 0: {}' .format(choices.count(1)/len(choices), choices.count(0)/len(choices)))

#model.save('tpy.model')
