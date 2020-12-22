"""Last modified on 12/2020
The EnvControl class controls the environment. It manages a plethora of different aspects of the environment and acts as an umpire in regards to the relationship between the car and the environment.
Some of the things, the class handles:
Server/Client relations (the car and the environment).
Spawning a car of model type tesla (The agent) at random location in carla environment [Every time it restarts].
Provides the means for manual control.
Attaching the camera to the front of the car and preproccessing the images it recieves into RGB format.
One step dynamics: Meaning it defines how the car should take a step in its environment through a governing body of rules.
Collision detection.
Destroys actors at the end of each episode.
"""

class EnvControl:
 
    # Full steering amount
    STEER_AMT = 0.7
    
    # The size of the frame.
    im_width = 500
    im_height = 500
    
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
        #  If there isnt a collision check the speed if all is well reward it
        if len(self.collision_hist) != 0:
            done = True
            reward = -100
        elif kmh < 35:
            done = False
            reward = -15
        elif kmh > 120:
            done = False
            reward = - 10
        else:
            done = False
            reward = 15
        # Is the episode over already ?
        if self.episode_start + self.SECONDS_PER_EPISODE < time.time():
            done = True
        # Getting the new input based, the reward and the terminal state boolean
        return self.front_camera, reward, done, None
