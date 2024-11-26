import gymnasium as gym
from gymnasium import spaces
import numpy as np
import carla
import time
import math
import torch

# Initial attempt to create env wrapper
# Sources:
# https://github.com/alberto-mate/CARLA-SB3-RL-Training-Environment
# https://github.com/gustavomoers/E2E-CARLA-ReinforcementLearning-PPO/tree/main
class CarlaEnv(gym.Env):
    def __init__(self, 
                 client, 
                 carla_world, 
                 hud,
                 fps: int = 20):

        self.world = carla_world
        self.client = client
        self.map = self.world.get_map()
        self.hud = hud
        
        self.fps = fps
        
        self.player = None
        self.camera_rgb = None
        
        self.steer = 0
        self._control = carla.VehicleControl()
        self.max_dist = 4.5

        self.last_v = 0
        self.last_y = 0

        self.episode_reward = 0
        self.episode_counter = 0
        
        self.controller = None
        self.control_count = 0.0
        
        self.counter = 0
        
        self.action_space = spaces.Box(low=-1, high=1,shape=(2,),dtype="float")
        self.observation_space = spaces.Box(low=-0, high=255, shape=(128, 128, 1), dtype=np.uint8)
        
        self.device = (torch.device("cuda:0") if torch.cuda.is_available() else "cpu",)
        self.dinov2 = torch.hub.load("facebookresearch/dinov2", "dinov2_vits14_reg")
        self.dinov2 = self.dinov2.to(self.device[0])
        
    def get_dino_features(self, image):
        if isinstance(image, np.ndarray):
            image_on_gpu = torch.tensor(image, device=self.device[0])
            if image_on_gpu.shape[0] != 224 and image_on_gpu.shape[1] != 224:
                image_on_gpu = self.resize_image(image_on_gpu)
        else:
            raise TypeError("unsuported type")
        chanels_third = image_on_gpu.permute((3, 2, 0, 1))
        # shape = chanels_third.shape
        # stacked_frames = image_on_gpu.reshape(shape=tuple([shape[0] * shape[1], *shape[2:]]))
        with torch.no_grad():
            result = self.dinov2.forward_features(chanels_third)
        patch_embedings: torch.Tensor = result["x_norm_patchtokens"]
        return patch_embedings.cpu().numpy()

    def reset(self):
        print("Resetting environment")
        self.world.tick()

        print("Destroying actors")
        for actor in [self.player, self.camera_rgb]:
            if actor is not None:
                try:
                    actor.destroy()
                    self.world.tick()
                except:
                    pass
        
        print("Setting up environment")
        settings = self.world.get_settings()
        settings.fixed_delta_seconds = 1 / self.fps
        settings.synchronous_mode = True
        settings.no_rendering_mode = False
        self.world.apply_settings(settings)
        
        self.episode_reward = 0
        self.episode_counter += 1
        
        self.create_actors()
        
        print("Calculating player initial state")
        velocity_vec = self.player.get_velocity()
        current_transform = self.player.get_transform()
        current_location = current_transform.location
        current_roration = current_transform.rotation
        current_x = current_location.x
        current_y = current_location.y

        # TODO: implement wrap_angle
        current_yaw = wrap_angle(current_roration.yaw)
        current_speed = math.sqrt(velocity_vec.x**2 + velocity_vec.y**2 + velocity_vec.z**2)
        frame, current_timestamp = self.hud.get_simulation_information()
        self.controller.update_values(current_x, current_y, current_yaw, current_speed, current_timestamp, frame)
        self.episode_start = time.time()
        
        self.world.tick()
        
        velocity_vec_st = self.player.get_velocity()
        current_speed = math.sqrt(velocity_vec_st.x**2 + velocity_vec_st.y**2 + velocity_vec_st.z**2)
   
        # TODO: implement tick to get camera
        image_rgb = self.synch_mode.tick(timeout=10.0)
        if image_rgb is not None:
            # TODO: implement process_img2
            img = process_img2(self, image_rgb)
                
        last_transform = self.player.get_transform()
        last_location = last_transform.location
        self.last_y = last_location.y
        self.last_v = current_speed
        print(current_speed)

        return {
            "image": img,
        }
        
    
    def step(self, action):
        self.reward = 0
        done = False
        cos_yaw_diff = 0
        dist = 0
        collision = 0
        lane = 0
        traveled = 0
        obs = {}

        if action is None:
            raise ValueError("Action cannot be None")

        self.counter += 1
        self.global_t += 1
        
        if self.apply_vehicle_control(action):
            return
        
        # TODO: implement tick to get camera
        image_rgb = self.synch_mode.tick(timeout=10.0)

        # TODO: implement get_reward_comp
        cos_yaw_diff, dist, collision, lane, traveled = self.get_reward_comp(self.player, self.spawn_waypoint, collision, lane)
        # TODO: implement reward_value
        self.reward = self.reward_value(cos_yaw_diff, dist, collision, lane, traveled)
        self.episode_reward += self.reward

        if image_rgb is not None:
            # TODO: implement process_img2
            image = process_img2(self, image_rgb)
            obs["image"] = image
            obs["vit_embeddings"] = self.get_dino_features(image)
        
        truncated = False

        # TODO: redo rewards
        if collision == 1:
            done=True
            print("Episode ended by collision")
        
        if lane == 1:
            self.reward -= 50

        velocity_vec_st = self.player.get_velocity()
        current_speed = math.sqrt(velocity_vec_st.x**2 + velocity_vec_st.y**2 + velocity_vec_st.z**2)
        if current_speed < 0.1:
            done=True
                
        return obs, self.reward, done, truncated, {}
    

    def create_actors(self):

        self.blueprint_library = self.world.get_blueprint_library()
        self.vehicle_blueprint = self.blueprint_library.filter('*vehicle*')

        # PLAYER
    
        spawn_location = carla.Location()
        spawn_location.x = float(self.args.spawn_x)
        spawn_location.y = float(self.args.spawn_y)
        self.spawn_waypoint = self.map.get_waypoint(spawn_location)
        spawn_transform = self.spawn_waypoint.transform
        spawn_transform.location.z = 1.0
        self.player = self.world.try_spawn_actor(self.vehicle_blueprint.filter('model3')[0], spawn_transform)
        self.world.tick()
        print('vehicle spawned')

        # Turn on position lights
        current_lights = carla.VehicleLightState.NONE
        current_lights |= carla.VehicleLightState.Position
        self.player.set_light_state(carla.VehicleLightState.Position)

        # CAMERA RGB

        self.rgb_cam = self.blueprint_library.find('sensor.camera.rgb')
        self.rgb_cam.set_attribute("image_size_x", f"{640}")
        self.rgb_cam.set_attribute("image_size_y", f"{480}")
        self.rgb_cam.set_attribute("fov", f"110")
        self.camera_rgb = self.world.spawn_actor(
            self.rgb_cam,
            carla.Transform(carla.Location(x=2, z=1), carla.Rotation(0,0,0)),
            attach_to=self.player)
        self.world.tick()
        
        # SYNCH MODE CONTEXT

        self.synch_mode = CarlaSyncMode(self.world, self.camera_rgb, self.lane_invasion, self.collision_sensor)

        # SPECTATOR

        spectator = self.world.get_spectator()
        transform = self.player.get_transform()
        spectator.set_transform(carla.Transform(transform.location + carla.Location(y=-10,z=28.5), carla.Rotation(pitch=-90)))
        self.world.tick()

        # CONTROLLER

        self.control_count = 0
        if self.control_mode == "PID":
            self.controller = PIDController.Controller()


    def apply_vehicle_control(self, action):

        self.steer = action[0]
        print(f'steer = {self.steer}')
        self.acceleration = action[1]
        print(f'acceleration = {self.acceleration}')

        self._control.steer = self.steer

        if self.acceleration < 0:
             self._control.brake = np.abs(self.acceleration)
             self._control.throttle = 0

        else:
            self._control.throttle = self.acceleration
            self._control.brake = 0

        print(self._control)    

        self.player.apply_control(self._control)
        self.control_count += 1