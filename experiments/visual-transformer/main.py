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

import sys
# setting path
sys.path.append('../..')
import utils
from vit_extractor import ViTFeatureExtractor


class DataCollector:
    def __init__(self, config: dict) -> None:
        self.seed = config["seed"]
        config["algorithm"]["learning_rate"] = float(
            config["algorithm"]["learning_rate"]
        )
        config["training"]["steps"] = int(float(config["training"]["steps"]))
        self.config = config

    def single_env(self, seed: int = None) -> gym.Env:
        # lidar data still retuned be env
        sim_config = self.config["environment"].copy()
        sim_config.update(
            {
                "image_observation": True,
                "vehicle_config": dict(image_source="main_camera"),
                "sensors": {"main_camera": ()},
                # "agent_policy": IDMPolicy,  # drive with IDM policy
                "image_on_cuda": False,  # no use for default policy and multiprocessing
                "window_size": tuple(self.config["environment"]["window_size"]),
                "start_seed": seed if seed else self.seed,
                "use_render": False,
                "show_interface": False,
                "show_logo": False,
                "show_fps": False,
                "crash_vehicle_done": False,
                "crash_object_done": False,
                "out_of_road_done": True,
                "on_continuous_line_done": False,
            }
        )
        env = utils.FixedSafeMetaDriveEnv(sim_config)
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

    def show_view(self, observations: np.ndarray | cp.ndarray) -> None:
        frames = observations["image"]
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
        total_samples = self.config["training"]["steps"]
        if True:
            env = self.create_env(seed)
            start_time = time.perf_counter()
            reset_time, obs = utils.measure_time(env.reset)
            print(f"Reset took: {reset_time}")
            for frame_index in range(total_samples):
                actions = np.array(
                    [
                        env.action_space.sample()
                        for _ in range(self.config["simulation"]["simulations_count"])
                    ]
                )
                env.step_async(actions)
                obs = env.step_wait()
                if self.config["simulation"]["show_view"]:
                    self.show_view(obs[0])
                frames.append(obs[:3])

            end_time = time.perf_counter()
            print("FPS:", frame_index / (end_time - start_time))
            print("Time elapsed:", end_time - start_time)
        return frames


def test_policy(policy_file:str) -> None:
    test_config = yaml.safe_load(open("configs/main.yaml", "r"))
    collector = DataCollector(test_config)
    model = PPO.load(policy_file)
    env = collector.create_env()
    obs = env.reset()
    obs["image"] = utils.resize(obs["image"])
    for _ in range(1000):
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, done, info = env.step(action)
        collector.show_view(obs)
        obs["image"] = utils.resize(obs["image"])
    env.close()

def main() -> None:
    config: dict = yaml.safe_load(open("./configs/main.yaml", "r"))
    print(f"Cores count: {os.cpu_count()}")

    random.seed(config["seed"])
    torch.manual_seed(config["seed"])
    np.random.seed(config["seed"])

    collector = DataCollector(config)
    
    
    policy_kwargs = dict(
        features_extractor_class=ViTFeatureExtractor,
    )
    model = PPO(
        policy="MultiInputPolicy",
        env=collector.create_env(),
        learning_rate=float(config["algorithm"]["learning_rate"]),
        n_steps=config["algorithm"]["batch_size"], #batch size, n_env*n_steps
        batch_size=config["algorithm"]["minibatch_size"], #minibatch size
        n_epochs = config["algorithm"]["n_epochs"], 
        gamma = config["algorithm"]["gamma"],
        gae_lambda=config["algorithm"]["gae_lambda"],
        policy_kwargs=policy_kwargs,
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
    # mlflow.log_param(key="policy_architecture", value=str(model.policy))
    # print(model.policy)

    # TODO: maybe create log file showing network architecture
    loggers = Logger(
        folder=None,
        output_formats=[HumanOutputFormat(sys.stdout), utils.MLflowOutputFormat()],
    )
    model.set_logger(loggers)

    # tags=tracking.get("tags", {})
    with mlflow.start_run(log_system_metrics=True) as run:
        flat_parameters_dict = utils.flat_dict(config)
        mlflow.log_params(flat_parameters_dict)

        model.learn(
            total_timesteps=int(float(config["training"]["steps"])),
            log_interval=1,
            progress_bar=True,
        )
    model.save("visual-transformer-policy-1")


if __name__ == "__main__":
    #TODO: why just 2 actions in env action space?
    main()
    sys.exit(0)
