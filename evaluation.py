import time
import numpy as np
import torch
import cv2
import cupy as cp

# Metadrive env info keys: 
# ['overtake_vehicle_num', 'velocity', 'steering', 'acceleration', 
# 'step_energy', 'episode_energy', 'policy', 
# 'navigation_command', 'navigation_forward', 'navigation_left', 'navigation_right', 
# 'action', 'raw_action', 
# 'crash_vehicle', 'crash_object', 'crash_building', 'crash_human', 'crash_sidewalk', 'out_of_road', 
# 'arrive_dest', 'max_step', 'env_seed', 'crash', 'step_reward', 'route_completion', 
# 'cost', 'total_cost', 'episode_reward', 'episode_length', 'episode', 'is_success', 
# 'TimeLimit.truncated', 'terminal_observation']

# Some are described here:
# https://metadrive-simulator.readthedocs.io/en/latest/reward_cost_done.html?highlight=velocity#step-information

# Energy calculated from here (its fuel consumption):
# https://github.com/metadriverse/metadrive/blob/b908149e422f2e7715207ca1eb81380342de5681/metadrive/component/vehicle/base_vehicle.py#L285

def model_configuration():
    model_config = {
        "models/chill-owl-867-1M": {
            "environment": {
                "window_size": [480, 480],
                "traffic_density": 0.15,
                "stack_size": 4,
                "accident_prob": 0.8,
                "map": "CCCCCCCCCCCCCCC",
                "num_scenarios": 200
            },
            "simulation": {
                "simulations_count": 1,
                "show_view": True
            },
            "evaluation": {
                "backbone": "vit",
                "obs_feature_key": "vit_embeddings",
                "num_episodes": 5,
                "just_embeddings": True,
                "show_view": False
            },
            "seed": 1
        },
        "models/classy-skink-104": {
            "environment": {
                "window_size": [480, 480],
                "traffic_density": 0.15,
                "stack_size": 4,
                "accident_prob": 0.8,
                "map": "CCCCCCCCCCCCCCC",
                "num_scenarios": 200
            },
            "simulation": {
                "simulations_count": 1,
                "show_view": True
            },
            "evaluation": {
                "backbone": "resnet",
                "obs_feature_key": "resnet_features",
                "num_episodes": 5,
                "just_embeddings": True,
                "show_view": False,
            },
            "seed": 1
        }
        # "models/chill-owl-867-main-cnn-8": {
        #     "environment": {
        #         "window_size": [480, 480],
        #         "traffic_density": 0.15,
        #         "stack_size": 4,
        #         "accident_prob": 0.8,
        #         "map": "CCCCCCCCCCCCCCC",
        #         "num_scenarios": 200
        #     },
        #     "simulation": {
        #         "simulations_count": 1,
        #         "show_view": True
        #     },
        #     "evaluation": {
        #         "backbone": "resnet",
        #         "obs_feature_key": "resnet_features",
        #         "num_episodes": 5,
        #         "just_embeddings": True,
        #         "show_view": False,
        #     },
        #     "seed": 1
        # },
    }

    return model_config

# TODO: maybe move to utils from main as well
def show_view(observations: np.ndarray | cp.ndarray) -> None:
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

def evaluate_trained_model(
    env,
    model,
    num_episodes=10,
    max_steps_per_episode=1000,
    just_embeddings=False,
    do_show_view=False,
    obs_feature_key="vit_embeddings"
):
    # Adjust the number of episodes based on the number of parallel environments
    effective_episodes = max(1, num_episodes // env.num_envs)

    print(f"Evaluating model for {effective_episodes} episodes...")

    # Metrics
    total_success_rate = 0
    total_distance_traveled = []
    total_velocities = []
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

        step_velocities = np.empty((env.num_envs,), dtype=object)
        for i in range(env.num_envs):
            step_velocities[i] = []

        episode_start_time = time.time()
       
        while not all(done) and np.any(step_count < max_steps_per_episode):
            # disables gradient calculation
            with torch.no_grad():
                if just_embeddings:
                    obs = {obs_feature_key: obs[obs_feature_key]}
                    # Predict the next action
                    action, _ = model.predict(obs, deterministic=True)

            obs, reward, done, info = env.step(action)
            
            if do_show_view:
                show_view(obs)
            
            for i in range(env.num_envs):
                if not done[i] and step_count[i] < max_steps_per_episode:
                    
                    # For debugging
                    if "reward" in info[i]:
                        print(f"Reward: {info[i]['reward']}")
                    if "acceleration" in info[i]:
                        print(f"Acceleration: {info[i]['acceleration']}")
                    if "velocity" in info[i]:
                        print(f"velocity: {info[i]['velocity']}")
                    if "position_delta" in info[i]:
                        print(f"Position delta: {info[i]['position_delta']}")
                    if "arrive_dest" in info[i]:
                        print(f"arrive_dest: {info[i]['arrive_dest']}")
                    if "max_step" in info[i]:
                        print(f"max_step: {info[i]['max_step']}")
                    if "collision" in info[i]:
                        print(f"collision: {info[i]['collision']}")
                    
                    total_reward[i] += reward[i]
                    collisions[i] += info[i].get('collision', 0)
                    step_velocities[i].append(info[i].get('velocity', 0))
                    step_count[i] += 1
    
        total_episode_time = time.time() - episode_start_time

        total_collisions += np.sum(collisions)
        total_steps.extend(step_count)
        total_rewards.extend(total_reward)
        total_episode_times.append(total_episode_time)

        episode_velocities = [np.zeros(env.num_envs)]
        for i in range(env.num_envs):
            # Check success condition from environment-specific info
            if info[i].get("arrive_dest", False):
                total_success_rate += 1
            
            env_velocity = np.mean(step_velocities[i])
            episode_velocities[i] = env_velocity
            
            # velocity is returned in km/h, so we convert time to hours
            total_episode_time_hours = total_episode_time / 3600
            distance_traveled[i] = env_velocity * total_episode_time_hours

        total_distance_traveled.extend(distance_traveled)
        total_velocities.extend(episode_velocities)

        print(f"Episode {episode + 1} completed in {total_episode_time} seconds")
        print(f"Episode time in hours: {total_episode_time / 3600}")
        print(f"Total reward: {total_reward}")
        print(f"Distance traveled: {distance_traveled}")
        print(f"Velocities: {episode_velocities}")
        print(f"Collisions: {collisions}")
        print(f"Steps: {step_count}")

        print(f"Is done: {done}")
        print(f"Total success rate: {total_success_rate}")

    # Compute metrics
    total_env_episodes = effective_episodes * env.num_envs
    success_rate = total_success_rate / total_env_episodes
    average_distance = np.mean(total_distance_traveled)
    collision_rate = total_collisions / total_env_episodes
    average_steps = np.mean(total_steps)
    average_inference_time = np.sum(total_episode_times) / np.sum(total_steps)
    average_speed = np.mean(total_velocities)
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


def evaluate_model(env, model, num_episodes=10, max_steps_per_episode=1000):
    # Adjust the number of episodes based on the number of parallel environments
    effective_episodes = max(1, num_episodes // env.num_envs)

    # Metrics
    total_success_rate = 0
    total_distance_traveled = []
    total_collisions = 0
    total_steps = []
    total_episode_times = []
    total_rewards = []

    for episode in range(effective_episodes):
        obs = env.reset()
        done = [False] * env.num_envs
        total_reward = np.zeros(env.num_envs)
        distance_traveled = np.zeros(env.num_envs)
        collisions = np.zeros(env.num_envs)
        step_count = np.zeros(env.num_envs)
        episode_start_time = time.time()

        while not all(done) and np.any(step_count < max_steps_per_episode):
            # Predict the next action
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, done, info = env.step(action)
            for i in range(env.num_envs):
                if not done[i] and step_count[i] < max_steps_per_episode:
                    total_reward[i] += reward[i]
                    # TODO: position_delta does not seem to be available in env step info
                    distance_traveled[i] += info[i].get('position_delta', 0)
                    collisions[i] += info[i].get('collision', 0)
                    step_count[i] += 1

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