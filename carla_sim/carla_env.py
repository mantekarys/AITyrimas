import os
import subprocess
import time
import pygame
from PIL import Image
from pygame.locals import *
from carla_sim.wrappers import *
import carla
import gymnasium as gym

# Source: https://github.com/alberto-mate/CARLA-SB3-RL-Training-Environment/blob/main/carla_env/envs/collect_data_manual_env.py

class CarlaDataCollector:
    def __init__(self, host="localhost", port=2000,
                 viewer_res=(1280, 720), obs_res=(224, 224),
                 num_images_to_save=12000,
                 num_init_image=10000,
                 output_dir="images", fps=30, action_smoothing=0.9):
        """
        A class for manually collecting image data from a running CARLA environment.

        Parameters:
            - host (str): IP address of the CARLA host
            - port (int): Port used to connect to CARLA
            - viewer_res (tuple[int, int]): Resolution of the spectator camera as a (width, height) tuple
            - obs_res (tuple[int, int]): Resolution of the observation camera as a (width, height) tuple
            - num_images_to_save (int): Maximum number of images to collect
            - output_dir (str): Directory in which to save collected images
            - fps (int): FPS of the client. If fps <= 0 then use unbounded FPS.
            - action_smoothing (float): Scalar used to smooth the incoming action signal. 1.0 = max smoothing, 0.0 = no smoothing
        """

        self.sensor_transforms = {
            "spectator": carla.Transform(carla.Location(x=-5.5, z=2.8), carla.Rotation(pitch=-15)),
            "dashboard": carla.Transform(carla.Location(x=1.6, z=1.7))
        }

        # Initialize pygame for visualization
        pygame.init()
        pygame.font.init()
        width, height = viewer_res
        if obs_res is None:
            out_width, out_height = width, height
        else:
            out_width, out_height = obs_res
        self.display = pygame.display.set_mode((width, height), pygame.HWSURFACE | pygame.DOUBLEBUF)
        self.clock = pygame.time.Clock()

        # Setup gym environment
        self.action_space = gym.spaces.Box(np.array([-1, 0]), np.array([1, 1]), dtype=np.float32)  # steer, throttle
        self.observation_space = gym.spaces.Box(low=0.0, high=1.0, shape=(*obs_res, 3), dtype=np.float32)
        self.fps = fps
        self.spawn_point = 1
        self.action_smoothing = action_smoothing

        self.done = False
        self.recording = False
        self.extra_info = []
        self.num_saved_observations = 0
        self.num_images_to_save = num_images_to_save
        self.num_init_image = num_init_image
        self.observation = {key: None for key in ["rgb", "segmentation"]}  # Last received observations
        self.observation_buffer = {key: None for key in ["rgb", "segmentation"]}
        self.viewer_image = self.viewer_image_buffer = None  # Last received image to show in the viewer

        self.output_dir = output_dir
        os.makedirs(os.path.join(self.output_dir, "rgb"), exist_ok=True)
        os.makedirs(os.path.join(self.output_dir, "segmentation"), exist_ok=True)

        self.autopilot = False
        self.world = None
        try:
            # Connect to carla
            self.client = carla.Client(host, port)
            self.client.set_timeout(60.0)

            # Create world wrapper
            self.world = World(self.client)

            settings = self.world.get_settings()
            settings.fixed_delta_seconds = 1 / self.fps
            settings.synchronous_mode = True
            self.world.apply_settings(settings)
            self.client.reload_world(False)  # reload map keeping the world settings

            # Get spawn location
            lap_start_wp = self.world.map.get_waypoint(carla.Location(x=-180.0, y=110))
            spawn_transform = lap_start_wp.transform
            spawn_transform.location += carla.Location(z=1.0)

            # Create vehicle and attach camera to it
            self.vehicle = Vehicle(self.world, self.world.map.get_spawn_points()[0])

            # Create cameras
            self.dashcam_rgb = Camera(self.world, out_width, out_height,
                                      transform=self.sensor_transforms["dashboard"],
                                      attach_to=self.vehicle, on_recv_image=lambda e: self._set_observation_image("rgb", e))
            # self.camera = Camera(self.world, width, height,
            #                      transform=self.sensor_transforms["spectator"],
            #                      attach_to=self.vehicle, on_recv_image=lambda e: self._set_viewer_image(e))

            tm = self.client.get_trafficmanager()
            tm_port = tm.get_port()
            self.vehicle.set_autopilot(self.autopilot, tm_port)
            tm.vehicle_percentage_speed_difference(self.vehicle.get_carla_actor(), -30)

            tm.ignore_lights_percentage(self.vehicle.get_carla_actor(), 100.0)
        except Exception as e:
            self.close()
            raise e

    def close(self):
        pygame.quit()
        if self.world is not None:
            self.world.destroy()
        self.closed = True

    def save_observation(self):
        # Blit image from spectator camera
        self.display.blit(pygame.surfarray.make_surface(self.viewer_image.swapaxes(0, 1)), (0, 0))

        # Superimpose current observation into top-right corner
        for i, (_, obs) in enumerate(self.observation.items()):
            obs_h, obs_w = obs.shape[:2]
            view_h, view_w = self.viewer_image.shape[:2]
            pos = (view_w - obs_w - 10, obs_h * i + 10 * (i + 1))
            self.display.blit(pygame.surfarray.make_surface(obs.swapaxes(0, 1)), pos)

        # Save current observations
        if self.recording and self.vehicle.get_speed() > 2.0:
            for obs_type, obs in self.observation.items():
                img = Image.fromarray(obs)
                img.save(
                    os.path.join(self.output_dir, obs_type, "{}.png".format(self.num_saved_observations + self.num_init_image)))
            self.num_saved_observations += 1
            if self.num_saved_observations >= self.num_images_to_save - self.num_init_image:
                self.done = True

        # Render to screen
        pygame.display.flip()

    def step(self, action):
        if self.is_done():
            raise Exception("Step called after CarlaDataCollector was done.")

        # Take action
        if action is not None:
            steer, throttle = [float(a) for a in action]
            # steer, throttle, brake = [float(a) for a in action]
            self.vehicle.control.steer = self.vehicle.control.steer * self.action_smoothing + steer * (
                    1.0 - self.action_smoothing)
            self.vehicle.control.throttle = self.vehicle.control.throttle * self.action_smoothing + throttle * (
                    1.0 - self.action_smoothing)

        # Tick game
        self.world.tick()
        self.clock.tick()

        # Get most recent observation and viewer image
        self.observation["rgb"] = self._get_observation("rgb")
        self.viewer_image = self._get_viewer_image()

        pygame.event.pump()
        keys = pygame.key.get_pressed()
        if keys[K_ESCAPE]:
            self.done = True
        if keys[K_SPACE]:
            self.recording = True

    def is_done(self):
        return self.done

    def _get_observation(self, name):
        while self.observation_buffer[name] is None:
            pass
        obs = self.observation_buffer[name].copy()
        self.observation_buffer[name] = None
        return obs

    def _get_viewer_image(self):
        while self.viewer_image_buffer is None:
            pass
        image = self.viewer_image_buffer.copy()
        self.viewer_image_buffer = None
        return image

    def _set_observation_image(self, name, image):
        self.observation_buffer[name] = image

    def _set_viewer_image(self, image):
        self.viewer_image_buffer = image
