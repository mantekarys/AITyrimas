import os
import random
import sys
import time
from functools import partial
from typing import Any, Callable, Dict, Tuple, Union

import cupy as cp
import cv2
import gymnasium as gym
import gymnasium.spaces
import mlflow
import numpy as np
import stable_baselines3 as sb3
import torch
import torch.nn as nn
import torch.optim
import tqdm
import yaml
from metadrive import MetaDriveEnv, SafeMetaDriveEnv
from metadrive.component.sensors.rgb_camera import RGBCamera
from metadrive.envs.base_env import BaseEnv
from metadrive.policy.idm_policy import IDMPolicy
from stable_baselines3 import PPO
from stable_baselines3.common.logger import HumanOutputFormat, KVWriter, Logger
from stable_baselines3.common.policies import ActorCriticPolicy
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from stable_baselines3.common.vec_env import SubprocVecEnv

import utils
from vit_policy import ViTActorCriticPolicy


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
                "norm_pixel": True,
                "discrete_action": True,
                "discrete_throttle_dim": 3,
                "discrete_steering_dim": 3,
                "horizon": 500,
                "sensors": {"main_camera": ()},
                # "agent_policy": IDMPolicy,  # drive with IDM policy
                "image_on_cuda": False, #no use for default policy and multiprocessing
                "window_size": tuple(self.config["environment"]["window_size"]),
                "start_seed": seed if seed else self.seed,
                "use_render": False,
                "show_interface": False,
                "show_logo": False,
                "show_fps": False,
                "crash_vehicle_done":False,
                "crash_object_done":False,
                "out_of_road_done":True,
                "on_continuous_line_done":False,
            }
        )
        env = SafeMetaDriveEnv(sim_config)
        return env

    def create_env(self, seed: int = None) -> gym.Env:
        envs_count = self.config["simulation"]["simulations_count"]
        seed = seed if seed else self.config["seed"]
        num_scenarios = self.config["environment"]["num_scenarios"]
        parallel_envs = SubprocVecEnv(
            [
                partial(self.single_env, seed + index * num_scenarios)
                for index in range(envs_count)
            ]
        )
        return parallel_envs

def main() -> None:
    config: dict = yaml.safe_load(open("../../configs/main.yaml", "r"))
    print(f"Cores count: {os.cpu_count()}")

    random.seed(config["seed"])
    torch.manual_seed(config["seed"])
    np.random.seed(config["seed"])

    collector = DataCollector(config)
    
    model = PPO(
        policy=ViTActorCriticPolicy,
        env=collector.create_env(),
        learning_rate=float(config["algorithm"]["learning_rate"]),
        n_steps=config["algorithm"]["batch_size"], #batch size, n_env*n_steps
        batch_size=config["algorithm"]["minibatch_size"], #minibatch size
        n_epochs = config["algorithm"]["n_epochs"], 
        gamma = config["algorithm"]["gamma"],
        gae_lambda=config["algorithm"]["gae_lambda"], 
        # clip_range=
        # clip_range_vf=
        # ent_coef=
        # vf_coef=
        # max_grad_norm=
        # stats_window_size=100,
        seed=config["seed"],
        device= "cuda" if torch.cuda.is_available() else "cpu",
        verbose=1,
    )
    print(model.policy)
    sys.exit(0)

    loggers = Logger(
        folder=None,
        output_formats=[HumanOutputFormat(sys.stdout), utils.MLflowOutputFormat()],
    )
    model.set_logger(loggers)
    #TODO: Add parameters logging to mlflow
    #TODO: Set mlflow experiments to differentiante test and real runs
    with mlflow.start_run():
        model.learn(total_timesteps=float(config["training"]["steps"]),
                     log_interval=1, progress_bar=True)
    model.save("visual-transformer-policy-1")


if __name__ == "__main__":
    #TODO: why just 2 actions in env action space?
    main()
    sys.exit(0)
