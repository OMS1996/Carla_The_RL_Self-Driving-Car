# -*- coding: utf-8 -*-
"""
Created on Wed Dec  2 19:15:23 2020

@author: omarm
"""

## GLOBAL VARIABLES


# The frame dimesions.
IM_WIDTH = 640
IM_HEIGHT = 480


# The minibatch size
minibatch_size = 16

# Predictions size
PREDICTION_BATCH_SIZE = 1

# The size of the training batch.
TRAINING_BATCH_SIZE = minibatch_size // 4


# CNN archietecture
MODEL_NAME = "morra"

# The fraction of the GPU that is going to be used.
#MEMORY_FRACTION = 0.5

MIN_REWARD = - 100 # -- revisit this

# The number of episodes (NOT STEPS)
EPISODES = 1_000

# The discount rate from the bellman equation
DISCOUNT = 0.99

# Whether or not we will use the network
# The probability of using the network increases over time
# However it won't go below 0.001
epsilon = 1
EPSILON_DECAY = 0.9997
MIN_EPSILON = 0.001

# Get rewards every
GET_REWARD_STATS_EVERY = 25