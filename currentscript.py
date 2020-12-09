"""
Last modified on on Thu Nov 19 12:40:15 2020

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

## GLOBAL VARIABLES


very_start = time.time()
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
MODEL_NAME = "VGG16_GlobalMax2DPool"


MIN_REWARD = - 100 # -- revisit this

# The number of episodes (NOT STEPS)
EPISODES = 5

# The discount rate from the bellman equation
DISCOUNT = 0.997

# Whether or not we will use the network
# The probability of using the network increases over time
# However it won't go below 0.001
epsilon = 1
EPSILON_DECAY = 0.9997
MIN_EPSILON = 0.001

# Get rewards every
GET_REWARD_STATS_EVERY = 5



class EnvControl:
 
    
    # Full steering amount
    STEER_AMT = 0.7
    
    # The size of the frame.
    im_width = 640
    im_height = 480
    
    # The values coming in from the front camera after being processed.
    front_camera = None
    
    def __init__(self):
        
        # starting the environement
        self.client = carla.Client("localhost", 2000)
        self.client.set_timeout(3.0)
        
        # Getting the world into a variable.
        # Getting the car from the blueprint object
        self.world = self.client.get_world()
        self.blueprint_library = self.world.get_blueprint_library()
        self.model_3 = self.blueprint_library.filter("model3")[0]
        # The duration of the episode
        self.SECONDS_PER_EPISODE = 15
    
    def RESTART(self):
        # recording collisions
        self.collision_hist = []
        # Recording the list of actors for later destruction
        self.actor_list = []   
        
        # Getting the spawn locations (there are 200 spawn locations)
        # Then creating the vehicle 
        # Finally storing it into the actor list for later destruction
        self.transform = random.choice(self.world.get_map().get_spawn_points())
        self.vehicle = self.world.spawn_actor(self.model_3, self.transform)
        self.actor_list.append(self.vehicle)
        
        # Finding the RGB senor blueprint.
        # Initializing it with necassary parameters.
        # With front view
        self.rgb_cam = self.blueprint_library.find('sensor.camera.rgb')
        self.rgb_cam.set_attribute("image_size_x", f"{self.im_width}")
        self.rgb_cam.set_attribute("image_size_y", f"{self.im_height}")
        self.rgb_cam.set_attribute("fov", f"110")
        
        ## Creating the first Sensor (Sensor No. 1)
        # Getting the car location
        # relative to the car location putting the camera in the front area
        # Attaching it and assigning it
        # Putting it into the actor list
        # All the data coming into that sensor, get the RGB version of it.
        transform = carla.Transform(carla.Location(x=2.5, z=0.7))
        self.sensor = self.world.spawn_actor(self.rgb_cam, transform, attach_to=self.vehicle)
        self.actor_list.append(self.sensor)
        self.sensor.listen(lambda data: self.process_img(data))

        
        
        # Applying control as stationary
        self.vehicle.apply_control(carla.VehicleControl(throttle=0.0, brake=0.0))
        time.sleep(4)
        
        ## Creating the second sensor (Sensor No. 2)
        # Getting the blueprint for it.
        # Putting into the actors list for later destruction.
        colsensor = self.blueprint_library.find("sensor.other.collision")
        self.colsensor = self.world.spawn_actor(colsensor, transform, attach_to=self.vehicle)
        self.actor_list.append(self.colsensor)
        
        # Be stand by till you recieve input  -----
        while self.front_camera is None:
            time.sleep(0.01)
            
        # getting the starting time of the episode.
        self.episode_start = time.time()
        
        # Putting the car in a stationary position
        self.vehicle.apply_control(carla.VehicleControl(throttle=0.0, brake=0.0))
        
        return self.front_camera # Returning what the sensor sees.
    
    # Book keeping
    def collision_data(self, event):
        self.collision_hist.append(event)
    
    #
    def process_img(self, image):
        raw = np.array(image.raw_data) # convert to an array
        reshaped_image = raw.reshape((self.im_height, self.im_width, 4)) # was flattened, so we're going to shape it.
        rgb_image = reshaped_image[:, :, :3] # remove the alpha
        cv2.imshow("", rgb_image)
        cv2.waitKey(1)
        self.front_camera = rgb_image

    def step(self, action):
        '''
        - ALL actions right,left,forward
        - handle for the observation, possible collision, and reward
        
        '''
        if action == 0:
            self.vehicle.apply_control(carla.VehicleControl(throttle = 0.7, steer = -1*self.STEER_AMT))
        elif action == 1:
            self.vehicle.apply_control(carla.VehicleControl(throttle = 1.0, steer= 0))
        elif action == 2:
            self.vehicle.apply_control(carla.VehicleControl(throttle = 0.7, steer = 1*self.STEER_AMT))
        #elif action == 3:
            #self.vehicle.apply_control(carla.VehicleControl(throttle = 0.1, steer = 0.0, reverse = True))
        
        # Getting the velocity
        v = self.vehicle.get_velocity()
        
        # Getting the resultant velocity
        kmh = int(3.6 * math.sqrt(v.x**2 + v.y**2 + v.z**2))
        
        ## The Reward Shaping
        # if there is a collision Then just end the episode
        #  If there isnt a collision check the speed if all is well reward it
        if len(self.collision_hist) != 0:
            done = True
            reward = -100
        elif kmh < 60:
            done = False
            reward = -10
        else:
            done = False
            reward = 10
        # Is the episode over already ?
        if self.episode_start + self.SECONDS_PER_EPISODE < time.time():
            done = True
        # Getting the new input based, the reward and the terminal state boolean
        return self.front_camera, reward, done, None


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
        # Try with pre-trained network -----
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

    # Memory fraction, used mostly when training multiple agents #----------------
    ##gpu_options = tf.compat.v1.GPUOptions(per_process_gpu_memory_fraction = MEMORY_FRACTION)
    ##configuration = tf.compat.v1.ConfigProto(gpu_options=gpu_options)
    ##backend(config = configuration)
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
                    action = np.random.randint(0, 3)
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
    
    from pprint import pprint
    
    print()
    pprint(full_stat)
    
    print()
    print()
    print("Time for the whole script to run is : ",total_time_from_start_to_finish)
    
    # Transfer the list of lists to pandas dataframe.
    df = pd.DataFrame(full_stat[1:], columns=full_stat[0])
    
    # Save the df
    df.to_csv()

                  
                    
                 

                 
                                       
     
                    
                    