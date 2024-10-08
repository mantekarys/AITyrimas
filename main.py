import os
import random
import sys
import time
from functools import partial
from typing import Any, Callable, Dict, Tuple, Union

import cupy as cp
import cv2
import gymnasium as gym
import mlflow
import numpy as np
import stable_baselines3 as sb3
import torch
import tqdm
import yaml
from metadrive import MetaDriveEnv, SafeMetaDriveEnv
from metadrive.component.sensors.rgb_camera import RGBCamera
from metadrive.envs.base_env import BaseEnv
from metadrive.policy.idm_policy import IDMPolicy
from stable_baselines3 import PPO
from stable_baselines3.common.logger import HumanOutputFormat, KVWriter, Logger
from stable_baselines3.common.vec_env import SubprocVecEnv


def measure_time(func: Callable, *args, **kwargs) -> tuple[float, Any]:
    start = time.perf_counter()
    result = func(*args, **kwargs)
    end = time.perf_counter()
    return (end - start, result)


class DataCollector:
    def __init__(self, config: dict) -> None:
        self.seed = config["seed"]
        self.config = config

    def single_env(self, seed:int = None)->gym.Env:
        # lidar data still retuned be env
        sim_config = self.config["environment"].copy()
        sim_config.update(
            {
                "image_observation": True,
                "vehicle_config": dict(image_source="main_camera"),
                "sensors": {"main_camera": ()},
                "agent_policy": IDMPolicy,  # drive with IDM policy
                "image_on_cuda": False, #no use for default policy and multiprocessing
                "window_size": tuple(self.config["environment"]["window_size"]),
                "start_seed": seed if seed else self.seed,
                "use_render": False,
                "show_interface": False,
                "show_logo": False,
                "show_fps": False,
            }
        )
        env = SafeMetaDriveEnv(sim_config)
        return env

    def create_env(self, seed: int = None) -> gym.Env:
        envs_count = self.config["simulation"]["simulations_count"]
        seed = seed if seed else self.config["seed"]
        parallel_envs = SubprocVecEnv(
            [
                partial(self.single_env, seed + index)
                for index in range(envs_count)
            ]
        )
        return parallel_envs

    def show_view(self, observations: np.ndarray | cp.ndarray) -> None:
        frames = observations[0]["image"]
        if len(frames.shape) == 4:
            image = frames[..., -1] * 255  # [0., 1.] to [0, 255]
        elif len(frames.shape) == 5:
            frames = frames[:, ..., -1]
            image = np.concatenate([f for f in frames], axis=1)
            image *= 255
        image = image.astype(np.uint8)
        # image: np.array = cp.asnumpy(image)

        cv2.imshow("frame", image)
        if cv2.waitKey(1) == ord("q"):
            return

    def collect_frames(self) -> np.ndarray | cp.ndarray:
        frames = []
        seed = self.config["seed"]
        total_samples = self.config["training"]["batch_size"]
        if True:
            env = self.create_env(seed)
            start_time = time.perf_counter()
            reset_time, obs = measure_time(env.reset)
            print(f"Reset took: {reset_time}")
            for frame_index in range(total_samples):
                actions = np.array(
                    [env.action_space.sample() for _ in range(self.config["simulation"]["simulations_count"])]
                )
                env.step_async(actions)
                obs = env.step_wait()
                if self.config["simulation"]["show_view"]:
                    self.show_view(obs)
                frames.append(obs[:3])

            end_time = time.perf_counter()
            print("FPS:", frame_index / (end_time - start_time))
            print("Time elapsed:", end_time - start_time)
        return frames


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


def main() -> None:
    config: dict = yaml.safe_load(open("configs/main.yaml", "r"))
    print(f"Cores count: {os.cpu_count()}")
    random.seed(config["seed"])
    torch.manual_seed(config["seed"])
    np.random.seed(config["seed"])
    collector = DataCollector(config)
    # frames = collector.collect_frames()

    loggers = Logger(
        folder=None,
        output_formats=[HumanOutputFormat(sys.stdout), MLflowOutputFormat()],
    )

    model = PPO(
        policy="MultiInputPolicy",
        env=collector.create_env(),
        learning_rate=1e-5,
        n_steps=4096, #batch size, n_env*n_steps
        batch_size=64, #minibatch size
        n_epochs = 10, 
        gamma = 0.99,
        gae_lambda=0.95, 
        # clip_range=
        # clip_range_vf=
        # ent_coef=
        # vf_coef=
        # max_grad_norm=
        stats_window_size=100,
        seed=config["seed"],
        device= "cuda" if torch.cuda.is_available() else "cpu",
        verbose=1,
    )
    model.set_logger(loggers)
    with mlflow.start_run():
        model.learn(total_timesteps=1e5, log_interval=4, progress_bar=True)
    model.save("policy_1")


if __name__ == "__main__":
    main()
    sys.exit(0)

