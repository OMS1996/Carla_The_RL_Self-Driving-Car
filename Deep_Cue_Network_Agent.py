"""
Created on Wed Dec  2 19:10:09 2020

@author: omarm
"""

# Imports
import math
import numpy as np
import pandas as pd
import cv2
from collections import deque
import tensorflow as tf
from keras.layers import Dense, GlobalMaxPool2D
from keras.optimizers import Adam
from keras.models import Model
from keras.applications.vgg16 import VGG16
import glob
import os
import sys
from tqdm import tqdm
import random
import time
from threading import Thread




# Getting the necassary files through glob
try:
    sys.path.append(glob.glob('../carla/dist/carla-*%d.%d-%s.egg' % (
        sys.version_info.major,
        sys.version_info.minor,
        'win-amd64' if os.name == 'nt' else 'linux-x86_64'))[0])
except IndexError:
    pass

# Importing the carla API Classes.
import carla

class Deep_Cue_Network_Agent:
    def __init__(self):
        # Target model and fitment model.
        self.model = self.build_model()
        self.target_model = self.build_model()
        self.target_model.set_weights(self.model.get_weights())
        
        # The size of the replay memory.
        self.REPLAY_MEMORY_SIZE = 4_000
        
        # Q - Learning specific variable
        # The replay memory, using a deque (A queue that can be used from both sides).
        self.replay_memory = deque(maxlen= self.REPLAY_MEMORY_SIZE)

        # The minimum replay memory size.
        self.MIN_REPLAY_MEMORY_SIZE = 1_000
        
        # State and network attributes
            # Tracker
            # termination boolean
            # Last episode that was logged in
            # Training flag
        self.target_update_counter = 0 # Update tracker
        self.terminate = False  # is it terminal state
        self.last_logged_episode = 0
        self.training_initialized = False
        
        # Update the target model every 5
        self.UPDATE_TARGET_EVERY = 5
        
        
        
        
    def build_model(self):
        
        
        # Creating the VGG16
        base_model = VGG16(include_top = False, weights = None, input_shape= (IM_HEIGHT,IM_WIDTH ,3))
        
        # getting the output 
        x = base_model.output
        
        # the pooling method
        x = GlobalMaxPool2D()(x)
        
        # Getting the predictions from the dense layer.
        predictions = Dense(3, activation="linear")(x)
        
        # setting the stage.
        model = Model(inputs = base_model.input, outputs=predictions)
        model.compile(loss="mse", optimizer=Adam(lr=0.001), metrics=["accuracy"])
        
        return model
    
    def update_replay_memory(self, transition):
        '''
         Funtion updates replay memory with the new frames.
         Input: transition -> Tuple :: (current_state, action, reward, new_state, done)
         Output: None
        '''
        self.replay_memory.append(transition)

    def train(self):
        
        if len(self.replay_memory) < self.MIN_REPLAY_MEMORY_SIZE:
            return
        
        print("Training started !!!!")
        
        # Getting a random sample from the replay memory relative to the minibatch size assigned.
        minibatch = random.sample(self.replay_memory, minibatch_size)
        
        # CURRENT Q - VALUES
        # Getting the current states from tuple which is at the 0th index
        # normalizing the frame and storing it in current states
        current_states = np.array([transition[0] for transition in minibatch])/255
        
        # Using them to predict
        current_qs_list = self.model.predict(current_states, PREDICTION_BATCH_SIZE)
            
        # New Q - values
        # Getting the new state which is at the 3rd index
        # Normalize the frames
        new_current_states = np.array([transition[3] for transition in minibatch])/255
        # FUTURE Q_LIST
        future_qs_list = self.target_model.predict(new_current_states, PREDICTION_BATCH_SIZE)

        # Change it into a Supervised learning problem, In somesense
        X = []
        y = []
        
        # Looping through the minibatch
        # Updating the q value if it is not over
        for index, (current_state, action, reward, new_state, done) in enumerate(minibatch):
            if not done:
                # The equation for updating the q-value
                new_q = reward + DISCOUNT * np.max(future_qs_list[index])
            else:
                new_q = reward
                
            # Update the current 
            current_qs = current_qs_list[index]
            current_qs[action] = new_q

            # X and Y like supervised learning.
            X.append(current_state)
            y.append(current_qs)
            
        # preparing the tensorboard
        log_this_step = False
        if current_episode_ptr > self.last_logged_episode:
            log_this_step = True
        self.last_log_episode = current_episode_ptr
        
        # Fitting the model
        self.model.fit(np.array(X)/255, np.array(y), batch_size = TRAINING_BATCH_SIZE, verbose = 0, shuffle = False)

        if log_this_step:
            self.target_update_counter += 1

        if self.target_update_counter > self.UPDATE_TARGET_EVERY:
            self.target_model.set_weights(self.model.get_weights())
            self.target_update_counter = 0
            
    def predict_qs(self, state): ## predict_qs for only one
        return self.model.predict(np.array(state).reshape(-1, *state.shape)/255)[0]
    
    def thread_loop(self):
        # toy data to warm up the model.
        X = np.random.uniform(size=(1, IM_HEIGHT, IM_WIDTH, 3)).astype(np.float32)
        y = np.random.uniform(size=(1, 3)).astype(np.float32)
        
        # Fitting the model
        self.model.fit(X,y, verbose = False, batch_size = 1) # Fitting the model of batch size one.
            
        # Initisalization setup
        self.training_initialized = True            
            
        # Infinite loop.
        while True:
            if self.terminate:
                return
            self.train()
            time.sleep(0.02)