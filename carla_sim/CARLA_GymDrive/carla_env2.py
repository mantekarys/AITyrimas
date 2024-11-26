import gymnasium as gym
from gymnasium.spaces import Box
from src.env.environment import CarlaEnv
import torch
import numpy as np
from torchvision import transforms


class CustomCarlaEnv(gym.Env):
    def __init__(self):
        self.env = CarlaEnv('carla-rl-gym-v0', time_limit=1000, initialize_server=False, random_weather=False, synchronous_mode=True, show_sensor_data=True, random_traffic=False)  # <-- Alternative way to create the environment

        self.device = (torch.device("cuda:0") if torch.cuda.is_available() else "cpu",)
        self.dinov2 = torch.hub.load("facebookresearch/dinov2", "dinov2_vits14_reg")
        self.dinov2 = self.dinov2.to(self.device[0])
        
        observations = {}
        
        # TODO: whats is state?
        # https://github.com/metadriverse/metadrive/blob/main/metadrive/obs/state_obs.py#L30
        # observations["state"] = self.env.observation_space["state"]
        
        observations["image"] = self.env.observation_space["rgb_data"]
        observations["vit_embeddings"] = Box(low=0, high=1, shape=(4, 256, 384))
        
        self.observation_space = gym.spaces.dict.Dict(spaces=observations)
        self.action_space = self.env.action_space
        
        self.render_mode = self.env.render_mode
        
        # frame buffer init with 4 empty frames (shape [480, 480, 3, 4])
        self.frame_buffer = np.zeros((360, 640, 3, 4), dtype=np.float32)
        self.episode = 0

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
    
    def resize_image(self, image: torch.Tensor) -> torch.Tensor:
        # image: torch.Size([480, 480, 3, 4])
        transformation = transforms.Resize(size=(224, 224))
        chanel_second = image.permute((3, 2, 0, 1))
        resized: torch.Tensor = transformation(chanel_second)
        original_order = resized.permute((2, 3, 1, 0))
        return original_order
    
    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        
        if "rgb_data" not in obs:
            raise ValueError("Image not found in observation")
        
        update_obs = {}
        # match pixel value types (uint8 to float32)
        image = obs["rgb_data"].astype(np.float32)
        self.frame_buffer = np.roll(self.frame_buffer, shift=-1, axis=3)
        self.frame_buffer[:, :, :, -1] = image

        update_obs["vit_embeddings"] = self.get_dino_features(self.frame_buffer)
        update_obs['image'] = self.frame_buffer
        
        # speed = obs["speed"]
        # acceleration = step_infos['acceleration']

        return update_obs, reward, terminated, truncated, info
    
    def reset(self):
        obs, info = self.env.reset()
        
        if "rgb_data" not in obs:
            raise ValueError("Image not found in observation")
        
        update_obs = {}
        # match pixel value types (uint8 to float32)
        image = obs["rgb_data"].astype(np.float32)
        self.frame_buffer = np.roll(self.frame_buffer, shift=-1, axis=3)
        print(image.shape)
        self.frame_buffer[:, :, :, -1] = image
        
        update_obs["vit_embeddings"] = self.get_dino_features(self.frame_buffer)
        update_obs['image'] = self.frame_buffer
        
        return update_obs
    
    def close(self):
        print("Closing environment")
        self.env.close()