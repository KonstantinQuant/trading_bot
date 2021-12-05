# -*- coding: utf-8 -*-

import numpy as np
from numpy import random
import random
import pandas as pd
import tensorflow as tf
from datetime import datetime
import itertools
import argparse
import re
import os
import pickle
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler 
from sklearn.preprocessing import MinMaxScaler
from collections import deque 
      
from keras.models import Sequential
from keras.layers import Dense

import matplotlib.pyplot as plt

def getData():
  '''
  retrieve the closing values of the stock

  '''
  msft = pd.read_csv('BP.csv')
  msft = msft.dropna()
  print(msft.head())
  return msft.iloc[:, 5].values.reshape(-1,1)

print(getData().reshape(-1,1).shape)
plt.plot(getData()[7000:])

"""# 1. Create the Environment 
 
The environment represents all the "world" where the agent resides and stores all the data based on which the agent makes a decision.
"""

class TradingEnvironment:

  def __init__(self, data, initial_money):
    self.stock_prices = data

    self.initial_money = initial_money
    self.shares_owned = None
    self.cash = None
    self.stock_price = None

    self.n_steps = data.shape[0]
    self.curr_step = None

    # Define 3 actions: Sell = 0, Hold = 1, Buy = 2
    self.actions = [0,1,2]

    self.reset()

  def get_state(self):
    state = np.zeros(3)
    state[0] = float(self.shares_owned)
    state[1] = float(self.stock_price)
    state[2] = float(self.cash)
    return state


  def reset(self):
    self.curr_step = 0
    self.shares_owned = 0
    self.stock_price = self.stock_prices[self.curr_step]
    self.cash = self.initial_money
    return self.get_state()
    

  def trade(self, action):

    # sell all shares
    if action == 0:
      self.cash = self.cash + self.stock_price * self.shares_owned
      self.shares_owned = 0
      #print("I sold all shares.")

    # buy shares until we run out of money
    if action == 2:
      liquid = True

      while liquid:
        if self.cash >= self.stock_price:
          self.shares_owned = self.shares_owned + 1
          self.cash = self.cash - self.stock_price # buy
          #print("I bought a share for:", self.stock_price, "we own", self.shares_owned, "shares.")
        else:
          liquid = False

      if action == 1:
        self.cash = self.cash * 1+(0.01/252) # for each day the agent chooses to not do anything, he earns the risk free interest rate

  def _get_balance(self):
    return self.shares_owned * self.stock_price + self.cash

  
  def make_step(self, action):

    balance = self._get_balance()

    self.curr_step = self.curr_step + 1

    self.stock_price = self.stock_prices[self.curr_step]

    self.trade(action)

    newbalance = self._get_balance()

    reward = newbalance - balance

    done = self.curr_step == self.n_steps-1

    print("Step:", self.curr_step)
    print("Old Balance", balance, " New Balance:", newbalance, "Reward:", reward)

    return self.get_state(), reward, done, newbalance

"""# 2. Create the Replay Buffer

We will probably just use a deque

# 3. Create the Agent 

(with deep neural network)
"""

# this deep neural network predicts the q-values (aka the expected value of future rewards after performing that action in the current state)
def model(obs_count, neurons=32, layers=2):
  model = Sequential()
  model.add(Dense(neurons, input_shape=(obs_count,), activation='relu'))
  for _ in range(layers):
    model.add(Dense(neurons, activation='relu'))
  model.add(Dense(3, activation='linear')) # 3, since we have 3 actions
  history = model.compile(optimizer = 'adam', loss = 'mean_squared_error', metrics=['mean_squared_error'])

  return model

class TradingAgent:

  def __init__(self, state_size):

    self.memory = deque(maxlen=2000)
    self.state_size = state_size


    # for the q-formula:
    self.gamma = 0.95 # discount rate, we want future rewards to be worth less 
    self.epsilon = 1.0 # exploration factor, see act function
    self.epsilon_min = 0.01
    self.epsilon_decay = 0.995 # 5 percent decay
    self.model = model(state_size)

  def update_replay_buffer(self, state, action, reward, next_state, done):
    #if len(state[1]) > 0:
    self.memory.append((state, action, reward, next_state, done))

    #print("memory", self.memory)


  def act(self, state):
    # we dont always want the model to do the same things it remembers, so choose a random action from time 
    # to time (especially during beginning, since we let epsilon decay) so that we can learn new things
    # and dont just perform things we know are good 
    if np.random.rand() <= self.epsilon:
      r_action = random.randrange(3)
      return r_action

    # calculate Q(s, a)
    state = state.reshape(1,3)
    #test = np.zeros((3,3)) # my model will take any matrix that is x, 3... why?
    act_values = self.model.predict(state) 
    
    action = np.argmax(act_values[0])
    #print("Im choosing action", action)
    return action  # returns action

  # the "train" function: use minibatch training to train the model
  # https://towardsdatascience.com/reinforcement-learning-w-keras-openai-dqns-1eed3a5338c
  def replay(self, batch_size=32):
    # Predict the values of the q-table (deep q-learning)

    if len(self.memory) < batch_size:
      return

    # Samples us batch_size times (state, action, reward, next_state, done)
    # because we want to actively learn (train) from what we have learned in the past
    minibatch = random.sample(self.memory, batch_size)


    states = np.array([tup[0] for tup in minibatch]) # Retrieves all states from the memory
    actions = np.array([tup[1] for tup in minibatch])
    rewards = np.array([tup[2] for tup in minibatch])
    next_states = np.array([tup[3] for tup in minibatch])
    done = np.array([tup[4] for tup in minibatch])

    # Q(s', a)
    # optimal q-values fulfill the Bellman equation
    next_states = next_states.reshape(batch_size,3)
    target = rewards + self.gamma * np.max(self.model.predict(next_states), axis=1)
    target[done] = rewards[done]


    # Q(s, a)
    # for each state, we attempt to the predict the q value of that state, since we do not/can not store the whole table
    #print(states.shape)
    states = states.reshape(batch_size, 3)
    target_f = self.model.predict(states)
    #target_f[range(batch_size), actions] = target

    # the labels are in this case the optimal q values from the bellman equation
    # run one training step
    self.model.fit(states, target_f, epochs=1, verbose=0)
    
    # lower the exploration rate
    if self.epsilon > self.epsilon_min:
      self.epsilon *= self.epsilon_decay

"""# 4. Data preparation

Includes: Scaling,...
"""

def get_scaler(env):
  # function to standardize values
  # returns scikit-learn's scaler object to scale the states
  states = []
  for _ in range(env.n_steps):
    action = np.random.choice(env.actions)
    state, reward, done, balance = env.make_step(action)
    states.append(state)
    if done:
      break

  scaler = StandardScaler()
  scaler.fit(states)
  return scaler

"""# 5. Play one episode functionality"""

def play_episode(agent, env, train, scaler, batch_size=32):
  state = env.reset()
  state = scaler.transform([state])
  done = False

  while not done:
    action = agent.act(state)

    next_state, reward, done, balance = env.make_step(action)
    next_state = scaler.transform([next_state])
    if train:
      agent.update_replay_buffer(state, action, reward, next_state, done)
      agent.replay(batch_size)
    state = next_state

  return balance

"""# 6. Train and Main"""

# todo: scale the state, not the data itself, otherwise we will just buy  for the scaled price...
data = getData()[7000:]

env = TradingEnvironment(data, 100) # Create the Environment

scaler = get_scaler(env)

state_dim = 1*2+1 # tbd
gordongecko = TradingAgent(state_dim)

# Adjust these as wished
EPISODES = 500 

for e in range(EPISODES):
  print("Episode:",e)
  portfolio = list()
  val = play_episode(gordongecko, env, True, scaler)
  portfolio.append(val)
