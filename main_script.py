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

# The two other scripts ( two classes)
import EnvControl
import Deep_Cue_Network_Agent


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

## GLOBAL VARIABLES
very_start = time.time()

# The frame dimesions.
IM_WIDTH = 500
IM_HEIGHT = 500

# The minibatch size
minibatch_size = 16

# Predictions size
PREDICTION_BATCH_SIZE = 1

# The size of the training batch.
TRAINING_BATCH_SIZE = minibatch_size // 4

# CNN archietecture
MODEL_NAME = "VGG16_GlobalMax2DPool"

# Lowest possible reward
MIN_REWARD = - 100 

# The number of episodes (NOT STEPS)
EPISODES = 500

# The discount rate from the bellman equation
DISCOUNT = 0.997

# Whether or not we will use the network
# The probability of using the network increases over time
# However it won't go below 0.001
epsilon = 1
EPSILON_DECAY = 0.997
MIN_EPSILON = 0.001

# Get rewards every
GET_REWARD_STATS_EVERY = 5


if __name__ == '__main__':

    # For stats
    ep_rewards = []
    ep_rewards.append(MIN_REWARD) # Starting out with a low reward
    full_stat = []
    
    # The head
    full_stat.append(['average_reward','Min_reward', 'max_reward', 'epsilon'])

    # For more repetitive results
    random.seed(40)
    np.random.seed(40)
    tf.random.set_seed(40) #https://www.tensorflow.org/versions/r2.0/api_docs/python/tf/random/set_seed

    gpus = tf.config.experimental.list_physical_devices('GPU')
    if gpus:
      try:
        tf.config.experimental.set_virtual_device_configuration(gpus[0], [tf.config.experimental.VirtualDeviceConfiguration(memory_limit= 1024*2)])
      except RuntimeError as e:
        print(e)

    # Create models folder
    if not os.path.isdir('models'):
        os.makedirs('models')

    # Create agent and environment
    agent = Deep_Cue_Network_Agent()
    env = EnvControl()           
    
    # Start training thread and wait for training to be initialized
    trainer_thread = Thread(target=agent.thread_loop, daemon=True)
    trainer_thread.start()
    while not agent.training_initialized:
        time.sleep(0.01)
            
    # Initialize predictions - forst prediction takes longer as of initialization that has to be done
    # It's better to do a first prediction then before we start iterating over episode steps
    agent.predict_qs(np.ones((env.im_height, env.im_width, 3)))

    ID = 0

    # Iterate over episodes
    for episode in tqdm(range(1, EPISODES + 1), ascii=True, unit='episodes'):

            env.collision_hist = []

            # Update tensorboard step every episode
            current_episode_ptr = episode

            # Restarting episode - RESTART episode reward and step number
            episode_reward = 0
            step = 1
            
            # RESTART environment and get initial state
            current_state = env.RESTART()
            
            # RESTART flag and start iterating until episode ends
            done = False
            episode_start = time.time()
            
            # the data frame
            # Create the pandas DataFrame 
            df = pd.DataFrame() 
            df['Statistics'] = ["average_reward","min_reward","max_reward","epsilon"]

            # Play for given number of seconds only
            while True:

                # This part stays mostly the same, the change is to query a model for Q values
                if np.random.random() > epsilon:
                    # Get action from Q table [Use the newtwork, use the .predict()]
                    action = np.argmax(agent.predict_qs(current_state))
                else:
                    # Get random action
                    action = np.random.randint(0, 4)
                    # This takes no time, so we add a delay matching 10 FPS (prediction above takes longer)
                    time.sleep(1/10)
                    
                # Take a step within the episode.
                # get all the information the new state and reward acquired and whether or not it is done.
                new_state, reward, done, _ = env.step(action)

                # Total reward per episode.
                episode_reward += reward

                # Update replay memory every step. 
                agent.update_replay_memory((current_state, action, reward, new_state, done))                   
                
                # Update the current state.
                current_state = new_state
                
                # Update the step, register a new step.
                step += 1
                
                # whether or not the episode is done.
                if done:
                    break       
                
            # Destroy all the actors objects at the end of the episode.
            for actor in env.actor_list:
                actor.destroy()      
                
            # Time to get the metadata, the statistics basically.
            # Append episode reward to a list and log stats (every given number of episodes)
            
            ep_rewards.append(episode_reward) # adding the current episode reward
            
            # Registiering the stats every certain amount of `episodes`, so we can keep track of it
            # We are getting the average reward, the minumum and the maxiumum for the last certain amount of `episodes`
            if not episode % GET_REWARD_STATS_EVERY or episode == 1:
                
                # Getting the stats
                average_reward = sum(ep_rewards[-GET_REWARD_STATS_EVERY:])/len(ep_rewards[-GET_REWARD_STATS_EVERY:])
                min_reward = min(ep_rewards[-GET_REWARD_STATS_EVERY:])
                max_reward = max(ep_rewards[-GET_REWARD_STATS_EVERY:])
                
                # Making use of this update.
                agent_stats = [average_reward,min_reward, max_reward, epsilon]
                
                print()
                print("---------------------------")
                print("episode",episode)
                print("average_reward:min_reward:max_reward:epsilon")
                # Print stats.
                print(agent_stats)
                print("---------------------------")
                print()
                full_stat.append(agent_stats)
                
                # Move on to the next.
                

                # Save model, but only when min reward is greater or equal a set value
                if min_reward >= MIN_REWARD:
                    agent.model.save(f'models/{MODEL_NAME}__{max_reward:_>10.2f}max_{average_reward:_>10.2f}avg_{min_reward:_>7.2f}min_{EPISODES:_>4.0f}episodes__{int(time.time())}.model')                 
                    
            # Decay epsilon
            if epsilon > MIN_EPSILON:
                epsilon *= EPSILON_DECAY
                epsilon = max(MIN_EPSILON, epsilon) 

    # Now that all the madness above is over let's end it it              
    # Set termination flag for training thread
    # Kill the thread
    # Save the final model with its timestamp next to it,
    agent.terminate = True
    trainer_thread.join() 
    agent.model.save(f'models/{MODEL_NAME}__{int(time.time())}.model')  
    
    # Time at the end of the script
    very_end = time.time()
    
    # The total time needed for the script to run
    total_time_from_start_to_finish = very_end - very_start
    
    from pprint import pprint # Making it look pretty.
    
    print()
    pprint(full_stat) # Displaying to console.
    
    print()
    print()
    total_time_from_start_to_finish = total_time_from_start_to_finish /(60*60)
    print("Time for the whole script to run is : ",total_time_from_start_to_finish)
    
    # Transfer the list of lists to pandas dataframe.
    df = pd.DataFrame(full_stat[1:], columns=full_stat[0])
    
    # Save the df
    df.to_csv(f'Carla_Metrics_AccEpsilon__{int(time.time())}.csv')
