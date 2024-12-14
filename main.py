import os
import random
import sys
import time
from functools import partial

import cupy as cp
import cv2
import gymnasium as gym
import mlflow
import numpy as np
import pandas as pd
import torch
import torch.optim
import yaml
from mlflow.entities.run import Run
from stable_baselines3 import PPO
from stable_baselines3.common.logger import HumanOutputFormat, Logger
from stable_baselines3.common.vec_env import SubprocVecEnv

import utils
from cnn_custom_policy import CustomResNetPolicy

from evaluation import evaluate_model, evaluate_trained_model, model_configuration

class DataCollector:
    def __init__(self, config: dict) -> None:
        self.seed = config["seed"]
        self.return_image = config["simulation"]["show_view"]
        
        if "algorithm" in config:
            config["algorithm"]["learning_rate"] = float(
                config["algorithm"]["learning_rate"]
            )
            
        if "training" in config:
            config["training"]["steps"] = int(float(config["training"]["steps"]))

        if "evaluation" in config:
            self.evaluated = True
            self.agent_backbone = config["evaluation"].get("backbone", "vit")
            print(f"Agent backbone: {self.agent_backbone}")
            
            if self.agent_backbone not in ["resnet", "vit"]:
                raise ValueError("Evaluation error: Invalid backbone for the agent set in config file")

        self.config = config
        self.maps = self.config.get("environment", {}).get("map", [])

    def single_env(self, seed: int = None) -> gym.Env:
        sim_config = self.config["environment"].copy()
        sim_config.update(
            {
                "speed_reward": 0.0,
                "use_lateral_reward": False,
                "out_of_road_penalty": 30,
                "image_observation": True,
                "vehicle_config": dict(image_source="main_camera"),
                "sensors": {"main_camera": ()},
                "image_on_cuda": False,
                "window_size": tuple(self.config["environment"]["window_size"]),
                "start_seed": seed if seed else self.seed,
                "use_render": False,
                "show_interface": True,
                "show_logo": False,
                "show_fps": False,
                "crash_vehicle_done": False,
                "crash_object_done": False,
                "out_of_route_done": True,
                "on_continuous_line_done": True,
            }
        )
        sim_config["map"] = random.choice(self.maps)

        # if evalutated pretrained model, use environment set in config
        if self.evaluated:
            if self.agent_backbone == "vit":
                env = utils.FixedSafeMetaDriveEnv(
                    return_image=self.return_image, env_config=sim_config
                )
            elif self.agent_backbone == "resnet":
                env = utils.CNN_FixedSafeMetaDriveEnv(
                    return_image=self.return_image, env_config=sim_config
                )
            else:
                raise ValueError("Evaluation error: Invalid backbone for the agent set in config file")
        else:
            # Change environment here for training
            env = utils.CNN_FixedSafeMetaDriveEnv(
                return_image=self.return_image, env_config=sim_config
            )

        # Modify the reset function to select a new map each time
        original_reset = env.reset

        def reset_with_random_map(*args, **kwargs):
            if self.maps:
                chosen_map = random.choice(self.maps)
                env.env.config["map"] = chosen_map
            return original_reset(*args, **kwargs)

        env.reset = reset_with_random_map
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

def metadrive_policy_test_collecting_metrics(config_file: str = "evaluate.yaml"):
    config = yaml.safe_load(open(f"configs/{config_file}", "r"))

    if "evaluation" not in config:
        raise ValueError(f"Config file {config_file} does not contain evaluation settings")
    
    if "policy_file" not in config["evaluation"]:
        raise ValueError(f"Config file {config_file} does not contain a policy file path")
    
    policy_file = config["evaluation"]["policy_file"]
    collector = DataCollector(config)
    
    model = PPO.load(policy_file)
    env = collector.create_env()

    metrics = evaluate_trained_model(
        env, 
        model, 
        num_episodes=config["evaluation"].get("num_episodes", 10),
        just_embeddings=config["evaluation"].get("just_embeddings", True),
        do_show_view=config["evaluation"].get("show_view", False),
        obs_feature_key=config["evaluation"].get("obs_feature_key", "vit_embeddings"),
    )

    # Model file name is set as the stage name, this is for single model evaluation
    metrics["Stage"] = policy_file.split("/")[-1].split(".")[0]
    df = pd.DataFrame([metrics])

    date = time.strftime("%Y-%m-%d-%H-%M-%S")
    df.to_csv(f"metrics/metadrive_policy_test_metrics_{date}.csv", index=False)
    utils.display_results(df)

def metadrive_policy_test_multiple_models():    
    model_config = model_configuration()
    results = []
    
    for model_file, model_config in model_config.items():
        collector = DataCollector(model_config)
        
        model = PPO.load(model_file)
        env = collector.create_env()

        metrics = evaluate_trained_model(
            env, 
            model, 
            num_episodes=model_config["evaluation"].get("num_episodes", 10),
            just_embeddings=model_config["evaluation"].get("just_embeddings", True),
            do_show_view=model_config["evaluation"].get("show_view", False),
            obs_feature_key=model_config["evaluation"].get("obs_feature_key", "vit_embeddings"),
        )

        metrics["Stage"] = model_file.split("/")[-1].split(".")[0]
        results.append(metrics)
        
        env.close()
    
    df = pd.DataFrame(results)

    date = time.strftime("%Y-%m-%d-%H-%M-%S")
    df.to_csv(f"metrics/metadrive_policy_test_metrics_{date}.csv", index=False)
    utils.display_results(df)


def main(config_file: str = "main.yaml", base_model: str | None = None) -> None:
    if not config_file:
        raise ValueError("No config file was given!")
    config: dict = yaml.safe_load(open("configs/" + config_file, "r"))
    tracking = config.pop("mlflow")

    # if "tracking_uri" in tracking:
    #     mlflow.set_tracking_uri(tracking["tracking_uri"])
    # mlflow.set_experiment(tracking["experiment_name"])

    print(f"Cores count: {os.cpu_count()}")
    random.seed(config["seed"])
    torch.manual_seed(config["seed"])
    np.random.seed(config["seed"])
    torch.set_float32_matmul_precision("high")
    torch.backends.cudnn.benchmark = True



    if config["algorithm"]["learning_rate_decay"]:
        lr = utils.linear_decay_schedule(float(config["algorithm"]["learning_rate"]))
    else:
        lr = config["algorithm"]["learning_rate"]

    if config["algorithm"]["clip_range_decay"]:
        clip = utils.linear_decay_schedule(float(config["algorithm"]["clip_range"]))
    else:
        clip = config["algorithm"]["clip_range"]

    if config["algorithm"]["clip_range_vf_decay"]:
        clip_vf = utils.linear_decay_schedule(float(config["algorithm"]["clip_range_vf"]))
    else:
        clip_vf = config["algorithm"]["clip_range_vf"]

    # Load stages and criteria from the config
    stage_criteria = config["stage_criteria"]
    stages = stage_criteria.keys()
    current_stage = 1
    model = None  # Initialize model as None to pass on the first run
    results = []  # Collect metrics for all stages

    while current_stage <= len(stages):
        print(f"Running Stage {current_stage}...")
        # Update config based on the current stage
        stage_config = config.copy()
        specific_stage_config = config["stage_config"].get(current_stage, {})
        stage_config["environment"].update(specific_stage_config)
        collector = DataCollector(stage_config)
        parallel_envs = collector.create_env()

        if model is None:
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
            if base_model:
                model.set_parameters(
                    load_path_or_dict=base_model,
                    device="cuda" if torch.cuda.is_available() else "cpu",
                )
        else:
            # Set the new environment for the existing model
            model.set_env(parallel_envs)

        # Set up logging
        loggers = Logger(
            folder=None,
            output_formats=[HumanOutputFormat(sys.stdout), utils.MLflowOutputFormat()],
        )
        model.set_logger(loggers)

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

        # Evaluate to determine if the agent should progress to the next stage
        metrics = evaluate_model(parallel_envs, model, num_episodes=10)
        metrics["Stage"] = current_stage
        results.append(metrics)
        success_rate = metrics.get("Success Rate", 0)
        collision_rate = metrics.get("Collision Rate per Episode", 1.0)

        criteria = stage_criteria[current_stage]
        progress = True
        if "success_rate" in criteria and success_rate < criteria["success_rate"]:
            progress = False
        if "collision_rate" in criteria and collision_rate > criteria["collision_rate"]:
            progress = False

        if progress:
            print(f"Stage {current_stage} criteria met, moving to next stage...")
            current_stage += 1
        else:
            print(f"Stage {current_stage} criteria not met, repeating stage...")

        parallel_envs.close()

    # Save results to DataFrame for future comparison
    df = pd.DataFrame(results)
    df.to_csv("curriculum_main_evaluation_results.csv", index=False)
    utils.display_results(df)


if __name__ == "__main__":

    # main("main.yaml")

    # main("test_1.yaml", "models/upset-asp-587.zip")
    # test_policy("models/sincere-ape-126.zip", 2000)
    # test_policy("models/chill-owl-867.zip", 2000, just_embeddings=True)

    # metadrive_policy_test_collecting_metrics(config_file="evaluate_1.yaml")
    metadrive_policy_test_multiple_models()


# different environments
# torch compile?
# add episode statistics only when episode actualy ended. In
