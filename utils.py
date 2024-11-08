import time
from typing import Any, Callable, Dict, Tuple, Union

import cupy as cp
import gymnasium as gym
import gymnasium.spaces.dict
import mlflow
import numpy as np

# import cv2
import torch
import torchvision
from gymnasium.spaces import Box, Space
from metadrive import MetaDriveEnv, SafeMetaDriveEnv
from stable_baselines3.common.logger import KVWriter
from torchvision import transforms


def resize(image: np.ndarray, target_size: Tuple[int, int] = (224)):
    transformation = transforms.Resize(size=target_size)
    image: torch.Tensor = torch.tensor(image)
    shape = image.shape
    concatenated = image.reshape(shape=tuple([*shape[:-2], shape[-2] * shape[-1]]))
    chanel_second = concatenated.permute((0, 3, 1, 2))
    resized_image: torch.Tensor = transformation(chanel_second)
    chanel_last = resized_image.permute((0, 2, 3, 1))
    primary_shape = chanel_last.reshape(
        shape=tuple([shape[0], target_size[0], target_size[1], 3, 4])
    )
    return primary_shape


class FixedSafeMetaDriveEnv(gym.Env):
    def __init__(
        self, return_image: bool = True, env_config: dict = None, *args, **kwargs
    ):
        self.env = SafeMetaDriveEnv(env_config, *args, **kwargs)
        self.return_image = return_image

        self.device = (torch.device("cuda:0") if torch.cuda.is_available() else "cpu",)
        self.dinov2 = torch.hub.load("facebookresearch/dinov2", "dinov2_vits14_reg")
        self.dinov2 = self.dinov2.to(self.device[0])

        observations_shapes = {}
        if self.return_image:
            observations_shapes["state"] = self.env.observation_space["state"]
            observations_shapes["image"] = self.env.observation_space["image"]
        observations_shapes["vit_embeddings"] = Box(low=0, high=1, shape=(4, 256, 384))
        self.observation_space = gymnasium.spaces.dict.Dict(spaces=observations_shapes)

        self.action_space = self.env.action_space
        self.render_mode = self.env.render_mode
        self.reward_range = getattr(self.env, "reward_range", None)
        self.spec = getattr(self.env, "spec", None)

    def get_dino_features(self, image: np.ndarray | cp.ndarray):
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
        transformation = transforms.Resize(size=(224, 224))
        chanel_second = image.permute((3, 2, 0, 1))
        resized: torch.Tensor = transformation(chanel_second)
        original_order = resized.permute((2, 3, 1, 0))
        return original_order

    def step_info_adapter(self, step_info: dict) -> dict:
        step_info["episode"] = {
            "l": step_info.get("episode_length", 0),
            "r": step_info.get("episode_reward", 0),
        }
        step_info["is_success"] = step_info.get("arrive_dest", False)
        return step_info

    def step(self, *args, **kwargs):
        obs, rewards, terminateds, truncateds, step_infos = self.env.step(
            *args, **kwargs
        )
        if self.return_image:
            obs["vit_embeddings"] = self.get_dino_features(obs["image"])
        else:
            obs = {"vit_embeddings": self.get_dino_features(obs["image"])}
        step_infos = self.step_info_adapter(step_infos)
        velocity = step_infos["velocity"]
        acceleration = step_infos['acceleration']
        if velocity < 1 and acceleration <= 0:
            rewards -=(1-velocity)*0.1
        elif velocity > 15 and acceleration > 0:
            rewards -= (velocity-15)/5
        # print(velocity)
        # print(f"Reward:  {rewards}")
        # print(f"Episode: {step_infos['episode_reward']}")
        return obs, rewards, terminateds, truncateds, step_infos

    def reset(self, *args, **kwargs):
        obs, step_infos = self.env.reset()
        if self.return_image:
            obs["vit_embeddings"] = self.get_dino_features(obs["image"])
        else:
            obs = {"vit_embeddings": self.get_dino_features(obs["image"])}
        step_infos = self.step_info_adapter(step_infos)
        return obs, step_infos

    def render(self) -> Any:
        """Renders the environment.

        Returns:
            The rendering of the environment, depending on the render mode
        """
        return self.env.render(mode=self.render_mode)

    def close(self):
        """Closes the environment."""
        self.env.close()

    def __str__(self):
        """Returns the wrapper name and the unwrapped environment string."""
        return f"<{type(self).__name__}{self.env}>"

    def __repr__(self):
        """Returns the string representation of the wrapper."""
        return str(self)


def linear_decay_schedule(lr_start: float, target_share:float=0.01) -> Callable[[float], float]:
    final_value=lr_start*target_share
    def schedule(progress_left: float) -> float:
        return (lr_start-final_value) * progress_left + final_value
    return schedule


def flat_dict(nested_dict: dict, prefix: str = "") -> dict:
    flat_level = {}
    for k, v in nested_dict.items():
        new_key_name = f"{prefix}/{k}".strip("/")
        if isinstance(v, dict):
            lower_level = flat_dict(v, prefix=new_key_name)
            flat_level.update(**lower_level)
        else:
            flat_level[new_key_name] = str(v)
    return flat_level


def measure_time(func: Callable, *args, **kwargs) -> tuple[float, Any]:
    start = time.perf_counter()
    result = func(*args, **kwargs)
    end = time.perf_counter()
    return (end - start, result)


class MLflowOutputFormat(KVWriter):
    """
    Dumps key/value pairs into MLflow's numeric format.
    """

    def write(
        self,
        key_values: Dict[str, Any],
        key_excluded: Dict[str, Union[str, Tuple[str, ...]]],
        step: int = 0,
    ) -> None:

        for (key, value), (_, excluded) in zip(
            sorted(key_values.items()), sorted(key_excluded.items())
        ):

            if excluded is not None and "mlflow" in excluded:
                continue

            if isinstance(value, np.ScalarType):
                if not isinstance(value, str):
                    mlflow.log_metric(key, value, step)
