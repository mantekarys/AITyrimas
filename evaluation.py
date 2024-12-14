import time
import numpy as np
import torch

def evaluate_trained_model(
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
            print(obs.keys())
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