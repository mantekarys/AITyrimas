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
from custom_policy_3 import CustomViTPolicy2


class DataCollector:
    def __init__(self, config: dict) -> None:
        self.seed = config["seed"]
        self.return_image = config["simulation"]["show_view"]
        config["algorithm"]["learning_rate"] = float(
            config["algorithm"]["learning_rate"]
        )
        config["training"]["steps"] = int(float(config["training"]["steps"]))
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
                "out_of_road_done": False,
                "on_continuous_line_done": True,
            }
        )
        sim_config["map"] = random.choice(self.maps)

        env = utils.FixedSafeMetaDriveEnv(
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

def metadrive_policy_test_collecting_metrics(
    policy_file: str,
    frames_count: int = 1000,
    config_file: str = "main.yaml",
    just_embeddings=False,
):
    config = yaml.safe_load(open(f"configs/{config_file}", "r"))
    collector = DataCollector(config)
    
    model = PPO.load(policy_file)
    env = collector.create_env()

    metrics = evaluate_model(env, model, num_episodes=10, just_embeddings=just_embeddings)
    print(metrics)
    df = pd.DataFrame(metrics)
    date = time.strftime("%Y-%m-%d-%H-%M-%S")
    df.to_csv(f"metrics/metadrive_policy_test_metrics_{date}.csv", index=False)
    utils.display_results(df)

def evaluate_model(
    env,
    model,
    num_episodes=10,
    max_steps_per_episode=1000,
    just_embeddings=False
):
    # Adjust the number of episodes based on the number of parallel environments
    effective_episodes = max(1, num_episodes // env.num_envs)

    print(f"Evaluating model for {effective_episodes} episodes...")

    # Metrics
    total_success_rate = 0
    total_distance_traveled = []
    total_collisions = 0
    total_steps = []
    total_episode_times = []
    total_rewards = []

    for episode in range(effective_episodes):
        print(f"Episode {episode + 1}/{effective_episodes}")

        obs = env.reset()
        done = [False] * env.num_envs
        total_reward = np.zeros(env.num_envs)
        distance_traveled = np.zeros(env.num_envs)
        collisions = np.zeros(env.num_envs)
        step_count = np.zeros(env.num_envs)
        episode_start_time = time.time()
       
        while not all(done) and np.any(step_count < max_steps_per_episode):
            with torch.no_grad():
                if just_embeddings:
                    obs = {"vit_embeddings": obs["vit_embeddings"]}
            # Predict the next action
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, done, info = env.step(action)

            for i in range(env.num_envs):
                if not done[i] and step_count[i] < max_steps_per_episode:
                    total_reward[i] += reward[i]
                    distance_traveled[i] += info[i].get('position_delta', 0)
                    collisions[i] += info[i].get('collision', 0)
                    step_count[i] += 1

        print(f"Episode {episode + 1} completed in {time.time() - episode_start_time} seconds")
        print(f"Total reward: {total_reward}")
        print(f"Distance traveled: {distance_traveled}")
        print(f"Collisions: {collisions}")
        print(f"Steps: {step_count}")
        
        episode_end_time = time.time()
        total_distance_traveled.extend(distance_traveled)
        total_collisions += np.sum(collisions)
        total_steps.extend(step_count)
        total_rewards.extend(total_reward)
        total_episode_times.append(episode_end_time - episode_start_time)

        # Check success condition from environment-specific info
        for i in range(env.num_envs):
            if info[i].get("arrive_dest", False):
                total_success_rate += 1
                
        print(f"Total success rate: {total_success_rate}")

    # Compute metrics
    total_env_episodes = effective_episodes * env.num_envs
    success_rate = total_success_rate / total_env_episodes
    average_distance = np.mean(total_distance_traveled)
    collision_rate = total_collisions / total_env_episodes
    average_steps = np.mean(total_steps)
    average_inference_time = np.sum(total_episode_times) / np.sum(total_steps)
    average_speed = average_distance / np.sum(total_episode_times)
    average_reward = np.mean(total_rewards)

    metrics = {
        "Success Rate": success_rate,
        "Average Distance Traveled": average_distance,
        "Collision Rate per Episode": collision_rate,
        "Average Steps per Episode": average_steps,
        "Average Inference Time per Step (seconds)": average_inference_time,
        "Average Speed (distance per second)": average_speed,
        "Average Reward per Episode": average_reward
    }

    return metrics


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
                policy=CustomViTPolicy2,
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
    # main("test_1.yaml")
    # main("test_1.yaml", "models/upset-asp-587.zip")
    # test_policy("models/sincere-ape-126.zip", 2000)
    # test_policy("models/upset-asp-587.zip", 2000, config="test_1.yaml", just_embeddings=True)
    metadrive_policy_test_collecting_metrics(
        "models/upset-asp-587.zip",
        2000,
        config_file="evaluate_1.yaml",
        just_embeddings=True
    )


# different environments
# torch compile?
# add episode statistics only when episode actualy ended. In
