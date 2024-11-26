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
from metadrive.component.map.base_map import BaseMap
from metadrive.component.sensors.rgb_camera import RGBCamera
from metadrive.envs.base_env import BaseEnv
from metadrive.policy.idm_policy import IDMPolicy
from mlflow.entities.run import Run
from stable_baselines3 import PPO
from stable_baselines3.common.logger import HumanOutputFormat, KVWriter, Logger
from stable_baselines3.common.policies import ActorCriticPolicy
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from stable_baselines3.common.vec_env import SubprocVecEnv

import utils
from cnn_custom_policy import CustomResNetPolicy

# from custom_policy_1 import CustomCNNPolicy
# from custom_policy_2 import CustomViTPolicy
# from custom_policy_3 import CustomViTPolicy2


class DataCollector:
    def __init__(self, config: dict) -> None:
        self.seed = config["seed"]
        self.return_image = config["simulation"]["show_view"]
        config["algorithm"]["learning_rate"] = float(
            config["algorithm"]["learning_rate"]
        )
        config["training"]["steps"] = int(float(config["training"]["steps"]))
        self.config = config

    def single_env(self, seed: int = None) -> gym.Env:
        sim_config = self.config["environment"].copy()
        sim_config.update(
            {
                # "on_lane_line_penalty": 0.0,#1.0
                # "steering_range_penalty": 0.0,#0.5
                "speed_reward": 0.0,
                "use_lateral_reward": False,
                "out_of_road_penalty": 30,
                "image_observation": True,
                "vehicle_config": dict(image_source="main_camera"),
                "sensors": {"main_camera": ()},
                # "agent_policy": IDMPolicy,  # drive with IDM policy
                "image_on_cuda": False,  # no use for default policy and multiprocessing
                "window_size": tuple(self.config["environment"]["window_size"]),
                "start_seed": seed if seed else self.seed,
                "use_render": False,
                "show_interface": True,
                "show_logo": False,
                "show_fps": False,
                "crash_vehicle_done": False,
                "crash_object_done": False,
                "out_of_road_done": True,
                "on_continuous_line_done": False,
            }
        )
        env = utils.CNN_FixedSafeMetaDriveEnv(
            return_image=self.return_image, env_config=sim_config
        )
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


def test_policy(
    policy_file: str,
    frames_count: int = 1000,
    config: str = "main.yaml",
    just_embeddings=False,
) -> None:
    test_config = yaml.safe_load(open(f"configs/{config}", "r"))
    collector = DataCollector(test_config)
    model = PPO.load(policy_file)
    env = collector.create_env()
    obs = env.reset()
    obs["image"] = utils.resize(obs["image"], (224, 224))
    for _ in range(frames_count):
        with torch.no_grad():
            if just_embeddings:
                obs = {"vit_embeddings": obs["vit_embeddings"]}
            action, _ = model.predict(obs, deterministic=True)
        obs, reward, done, info = env.step(action)
        collector.show_view(obs)
        obs["image"] = utils.resize(obs["image"], (224, 224))
    env.close()


def main(config_file: str = "main.yaml", base_model: str | None = None) -> None:
    if not config_file:
        raise ValueError("No config file was given!")
    config: dict = yaml.safe_load(open("configs/" + config_file, "r"))
    tracking = config.pop("mlflow")
    if "tracking_uri" in tracking:
        mlflow.set_tracking_uri(tracking["tracking_uri"])
    mlflow.set_experiment(tracking["experiment_name"])
    print(f"Cores count: {os.cpu_count()}")
    random.seed(config["seed"])
    torch.manual_seed(config["seed"])
    np.random.seed(config["seed"])
    torch.set_float32_matmul_precision("high")

    if config["algorithm"]["learning_rate_decay"]:
        lr = utils.linear_decay_schedule(float(config["algorithm"]["learning_rate"]))
    else:
        lr = config["algorithm"]["learning_rate"]

    if config["algorithm"]["clip_range_decay"]:
        clip = utils.linear_decay_schedule(float(config["algorithm"]["clip_range"]))
    else:
        clip = config["algorithm"]["clip_range"]

    if config["algorithm"]["clip_range_vf_decay"]:
        clip_vf = utils.linear_decay_schedule(
            float(config["algorithm"]["clip_range_vf"])
        )
    else:
        clip_vf = config["algorithm"]["clip_range_vf"]

    collector = DataCollector(config)
    parallel_envs = collector.create_env()
    model = PPO(
        policy=CustomResNetPolicy,
        env=parallel_envs,
        learning_rate=lr,
        n_steps=config["algorithm"]["batch_size"],  # batch size, n_env*n_steps
        batch_size=config["algorithm"]["minibatch_size"],  # minibatch size
        n_epochs=config["algorithm"]["n_epochs"],
        gamma=config["algorithm"]["gamma"],
        gae_lambda=config["algorithm"]["gae_lambda"],
        clip_range=clip,
        clip_range_vf=clip_vf,
        ent_coef=config["algorithm"]["ent_coef"],
        vf_coef=config["algorithm"]["vf_coef"],
        max_grad_norm=config["algorithm"]["max_grad_norm"],
        stats_window_size=10,
        seed=config["seed"],
        device="cuda" if torch.cuda.is_available() else "cpu",
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

    if base_model:
        model.set_parameters(
            load_path_or_dict=base_model,
            device="cuda" if torch.cuda.is_available() else "cpu",
        )
    tags = tracking.get("tags", {})
    with mlflow.start_run(log_system_metrics=True, tags=tags) as run:
        flat_parameters_dict = utils.flat_dict(config)
        mlflow.log_params(flat_parameters_dict)

        model.learn(
            total_timesteps=int(float(config["training"]["steps"])),
            log_interval=1,
            progress_bar=True,
        )
        run: Run
        run_name = run.info.run_name
        model.save(os.path.join("models", run_name))
    parallel_envs.close()


if __name__ == "__main__":
    # main("test_3.yaml")
    # main("main.yaml")
    # main("test_1.yaml", "models/upset-asp-587.zip")
    # test_policy("models/sincere-ape-126.zip", 2000)
    test_policy(
        "models/chill-owl-867-big-cnn-4.zip", 2000, config="test_1.yaml", just_embeddings=False
    )


# different environments
# torch compile?
# add episode statistics only when episode actualy ended. In
