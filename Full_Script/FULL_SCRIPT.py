"""Last modified on Wed Aug 28 2024
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
import logging

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Getting the necessary files through glob
try:
    sys.path.append(glob.glob('../carla/dist/carla-*%d.%d-%s.egg' % (
        sys.version_info.major,
        sys.version_info.minor,
        'win-amd64' if os.name == 'nt' else 'linux-x86_64'))[0])
except IndexError:
    logger.error("CARLA egg file not found. Please check your CARLA installation.")
    sys.exit(1)

# Importing the carla API Classes.
import carla

## GLOBAL VARIABLES
very_start = time.time()

# The frame dimensions.
IM_WIDTH = 500
IM_HEIGHT = 500

# The minibatch size
minibatch_size = 16

# Predictions size
PREDICTION_BATCH_SIZE = 1

# The size of the training batch.
TRAINING_BATCH_SIZE = minibatch_size // 4

# CNN architecture
MODEL_NAME = "VGG16_GlobalMax2DPool"

# Lowest possible reward
MIN_REWARD = -100 

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

class EnvControl:
    # Full steering amount
    STEER_AMT = 0.7

    # The size of the frame.
    im_width = 500
    im_height = 500

    # The values coming in from the front camera after being processed.
    front_camera = None

    def __init__(self):
        # starting the environment
        self.client = carla.Client("localhost", 2000)
        self.client.set_timeout(3.0)

        # Getting the world into a variable.
        # Getting the car from the blueprint object
        self.world = self.client.get_world()
        self.blueprint_library = self.world.get_blueprint_library()
        self.model_3 = self.blueprint_library.filter("model3")[0]
        # The duration of the episode
        self.SECONDS_PER_EPISODE = 10

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

        # Finding the RGB sensor blueprint.
        # Initializing it with necessary parameters.
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

        # Be stand by till you receive input
        while self.front_camera is None:
            time.sleep(0.01)

        # getting the starting time of the episode.
        self.episode_start = time.time()

        # Putting the car in a stationary position
        self.vehicle.apply_control(carla.VehicleControl(throttle=0.0, brake=0.0))

        return self.front_camera # Returning what the sensor sees.

    def process_img(self, image):
        raw = np.array(image.raw_data) # convert to an array
        reshaped_image = raw.reshape((self.im_height, self.im_width, 4)) # was flattened, so we're going to shape it.
        rgb_image = reshaped_image[:, :, :3] # remove the alpha
        cv2.imshow("", rgb_image)
        cv2.waitKey(1)
        self.front_camera = rgb_image

    # Book keeping
    def collision_data(self, event):
        self.collision_hist.append(event)

    def step(self, action):
        '''
        - ALL actions right,left,forward
        - handle for the observation, possible collision, and reward
        '''
        if action == 0:
            self.vehicle.apply_control(carla.VehicleControl(throttle = 0.7, steer = - 1 * self.STEER_AMT)) # left
        elif action == 1:
            self.vehicle.apply_control(carla.VehicleControl(throttle = 0.5, steer = 0.0)) # Half-throttle
        elif action == 2:
            self.vehicle.apply_control(carla.VehicleControl(throttle = 1.0, steer= 0)) # Full-throttle
        elif action == 3:
            self.vehicle.apply_control(carla.VehicleControl(throttle = 0.7, steer = 1 * self.STEER_AMT)) # right

        # Getting the velocity
        v = self.vehicle.get_velocity()

        # Getting the resultant velocity meter/sec
        kmh = int(3.6 * math.sqrt(v.x**2 + v.y**2 + v.z**2))

        ## The Reward Shaping
        # if there is a collision Then just end the episode
        #  If there isn't a collision check the speed if all is well reward it
        if len(self.collision_hist) != 0:
            done = True
            reward = -100
        elif kmh < 35:
            done = False
            reward = -15
        elif kmh > 120:
            done = False
            reward = -10
        else:
            done = False
            reward = 15
        # Is the episode over already?
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
        self.replay_memory = deque(maxlen=self.REPLAY_MEMORY_SIZE)

        # The minimum replay memory size.
        self.MIN_REPLAY_MEMORY_SIZE = 1_000

        # State and network attributes
        # Tracker
        # termination boolean
        # Last episode that was logged in
        # Training flag
        self.target_update_counter = 0  # Update tracker
        self.terminate = False  # is it terminal state
        self.last_logged_episode = 0
        self.training_initialized = False

        # Update the target model every 5
        self.UPDATE_TARGET_EVERY = 5

    def build_model(self):
        # Creating the VGG16
        # Try with pre-trained network
        base_model = VGG16(include_top=False, weights=None, input_shape=(IM_HEIGHT, IM_WIDTH, 3))

        # getting the output 
        x = base_model.output

        # the pooling method
        x = GlobalMaxPool2D()(x)

        # Getting the predictions from the dense layer.
        predictions = Dense(3, activation="linear")(x)

        # setting the stage.
        model = Model(inputs=base_model.input, outputs=predictions)
        model.compile(loss="mse", optimizer=Adam(lr=0.009), metrics=["accuracy"])

        return model

    def update_replay_memory(self, transition):
        '''
         Function updates replay memory with the new frames.
         Input: transition -> Tuple :: (current_state, action, reward, new_state, done)
         Output: None
        '''
        self.replay_memory.append(transition)

    def train(self):
        if len(self.replay_memory) < self.MIN_REPLAY_MEMORY_SIZE:
            return

        logger.info("Training started!")

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

        # Change it into a Supervised learning problem, In some sense
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
        self.model.fit(np.array(X)/255, np.array(y), batch_size=TRAINING_BATCH_SIZE, verbose=0, shuffle=False)

        if log_this_step:
            self.target_update_counter += 1

        if self.target_update_counter > self.UPDATE_TARGET_EVERY:
            self.target_model.set_weights(self.model.get_weights())
            self.target_update_counter = 0

    def predict_qs(self, state):  ## predict_qs for only one
        return self.model.predict(np.array(state).reshape(-1, *state.shape)/255)[0]

    def thread_loop(self):
        # toy data to warm up the model.
        X = np.random.uniform(size=(1, IM_HEIGHT, IM_WIDTH, 3)).astype(np.float32)
        y = np.random.uniform(size=(1, 3)).astype(np.float32)

        # Fitting the model
        self.model.fit(X, y, verbose=False, batch_size=1)  # Fitting the model of batch size one.

        # Initialization setup
        self.training_initialized = True            

        # Infinite loop.
        while True:
            if self.terminate:
                return
            self.train()
            time.sleep(0.02)

def setup_gpu():
    gpus = tf.config.experimental.list_physical_devices('GPU')
    if gpus:
        try:
            tf.config.experimental.set_virtual_device_configuration(
                gpus[0], 
                [tf.config.experimental.VirtualDeviceConfiguration(memory_limit=2048)]
            )
        except RuntimeError as e:
            logger.error(f"GPU setup error: {e}")

def create_models_folder():
    if not os.path.isdir('models'):
        os.makedirs('models')

def choose_action(agent, state, epsilon):
    if np.random.random() > epsilon:
        return np.argmax(agent.predict_qs(state))
    else:
        return np.random.randint(0, 4)

def run_single_episode(agent, env, epsilon):
    env.collision_hist = []
    current_state = env.RESTART()
    episode_reward = 0
    step = 1
    done = False
    episode_start = time.time()

    while True:
        action = choose_action(agent, current_state, epsilon)
        new_state, reward, done, _ = env.step(action)
        episode_reward += reward
        agent.update_replay_memory((current_state, action, reward, new_state, done))
        current_state = new_state
        step += 1

        if done:
            break

    for actor in env.actor_list:
        actor.destroy()

    return episode_reward


def calculate_and_log_stats(ep_rewards, epsilon):
    recent_rewards = ep_rewards[-GET_REWARD_STATS_EVERY:]
    average_reward = sum(recent_rewards) / len(recent_rewards)
    min_reward = min(recent_rewards)
    max_reward = max(recent_rewards)
    stats = [average_reward, min_reward, max_reward, epsilon]

    logger.info(f"Episode stats - Avg Reward: {average_reward:.2f}, Min Reward: {min_reward:.2f}, Max Reward: {max_reward:.2f}, Epsilon: {epsilon:.4f}")
    return stats

def update_epsilon(epsilon):
    if epsilon > MIN_EPSILON:
        epsilon *= EPSILON_DECAY
        epsilon = max(MIN_EPSILON, epsilon)
    return epsilon

def save_model(agent, stats, episode):
    model_name = f'models/{MODEL_NAME}__{stats[2]:_>10.2f}max_{stats[0]:_>10.2f}avg_{stats[1]:_>7.2f}min_{episode:_>4.0f}episodes__{int(time.time())}.model'
    agent.model.save(model_name)
    logger.info(f"Model saved: {model_name}")

def main():
    global current_episode_ptr

    # For stats
    ep_rewards = [MIN_REWARD]  # Starting out with a low reward
    full_stat = [['average_reward', 'Min_reward', 'max_reward', 'epsilon']]

    # For more repetitive results
    random.seed(40)
    np.random.seed(40)
    tf.random.set_seed(40)

    setup_gpu()
    create_models_folder()

    # Create agent and environment
    agent = Deep_Cue_Network_Agent()
    env = EnvControl()           

    # Start training thread and wait for training to be initialized
    trainer_thread = Thread(target=agent.thread_loop, daemon=True)
    trainer_thread.start()
    while not agent.training_initialized:
        time.sleep(0.01)

    # Initialize predictions - first prediction takes longer as of initialization that has to be done
    # It's better to do a first prediction then before we start iterating over episode steps
    agent.predict_qs(np.ones((env.im_height, env.im_width, 3)))

    epsilon = 1  # Reset epsilon for the main loop

    # Iterate over episodes
    for episode in tqdm(range(1, EPISODES + 1), ascii=True, unit='episodes'):
        current_episode_ptr = episode
        episode_reward = run_single_episode(agent, env, epsilon)
        ep_rewards.append(episode_reward)

        if not episode % GET_REWARD_STATS_EVERY or episode == 1:
            stats = calculate_and_log_stats(ep_rewards, epsilon)
            full_stat.append(stats)

            if stats[1] >= MIN_REWARD:
                save_model(agent, stats, episode)

        epsilon = update_epsilon(epsilon)

    # Now that all the madness above is over let's end it
    # Set termination flag for training thread
    # Kill the thread
    # Save the final model with its timestamp next to it
    agent.terminate = True
    trainer_thread.join() 
    agent.model.save(f'models/{MODEL_NAME}__{int(time.time())}.model')  

    # Time at the end of the script
    very_end = time.time()

    # The total time needed for the script to run
    total_time_from_start_to_finish = (very_end - very_start) / (60 * 60)

    logger.info(f"Full stats: {full_stat}")
    logger.info(f"Time for the whole script to run: {total_time_from_start_to_finish:.2f} hours")

    # Transfer the list of lists to pandas dataframe.
    df = pd.DataFrame(full_stat[1:], columns=full_stat[0])

    # Save the df
    csv_filename = f'Carla_Metrics_AccEpsilon__{int(time.time())}.csv'
    df.to_csv(csv_filename)
    logger.info(f"Metrics saved to {csv_filename}")

if __name__ == '__main__':
    main()
