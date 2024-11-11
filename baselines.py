import time
import yaml
import pandas as pd
import numpy as np
import torch
import matplotlib.pyplot as plt
import seaborn as sns
from stable_baselines3 import PPO, A2C, SAC, TD3
import utils
from main import DataCollector
import gc  # Import garbage collection module


# General training and evaluation function
def train_and_evaluate_baseline(algorithm_class, algorithm_name, collector, config, model=None, total_timesteps=1000,
								num_episodes=10):
	env = collector.create_env()
	try:
		if model is None:
			model = algorithm_class(
				policy="MultiInputPolicy",
				env=env,
				learning_rate=utils.linear_decay_schedule(float(config["algorithm"]["learning_rate"])),
				n_steps=config["algorithm"]["batch_size"],
				batch_size=config["algorithm"]["minibatch_size"],
				n_epochs=config["algorithm"]["n_epochs"],
				gamma=config["algorithm"]["gamma"],
				gae_lambda=config["algorithm"]["gae_lambda"],
				seed=config["seed"],
				device="cuda" if torch.cuda.is_available() else "cpu",
				verbose=1,
			)
		model.learn(total_timesteps=total_timesteps)
		metrics = evaluate_model(env, model, num_episodes=num_episodes)
		metrics["Algorithm"] = algorithm_name
		# Clear memory related to the model to avoid memory leaks
		torch.cuda.empty_cache()  # Free up GPU memory
		gc.collect()  # Run garbage collection to free memory
		return metrics, model
	finally:
		env.close()


# Evaluation function
def evaluate_model(env, model, num_episodes=10, max_steps_per_episode=1000):
	# Metrics
	total_success_rate = 0
	total_distance_traveled = []
	total_collisions = 0
	total_steps = []
	total_episode_times = []
	total_rewards = []

	for episode in range(num_episodes):
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

		# Check success condition from environment-specific info
		for i in range(env.num_envs):
			if info[i].get("arrive_dest", False):
				total_success_rate += 1

	total_episode_times.append(episode_end_time - episode_start_time)
	# Compute metrics
	success_rate = total_success_rate / (num_episodes * env.num_envs)
	average_distance = np.mean(total_distance_traveled)
	collision_rate = total_collisions / (num_episodes * env.num_envs)
	average_steps = np.mean(total_steps)
	average_inference_time = np.mean(total_episode_times) / np.mean(total_steps)
	average_speed = average_distance / np.mean(total_episode_times)
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


# Run baselines for multiple algorithms with curriculum learning
def run_all_baselines_with_curriculum():
	config = yaml.safe_load(open("configs/main.yaml", "r"))
	results = []

	# Run each algorithm across curriculum stages
	algorithms = [(PPO, "PPO"), (A2C, "A2C"), (SAC, "SAC"), (TD3, "TD3")]
	stages = [1, 2, 3]
	stage_criteria = {
		1: {"success_rate": 0.6, "evaluation_window": 10},
		2: {"success_rate": 0.7, "evaluation_window": 20},
		3: {"collision_rate": 0.15, "evaluation_window": 30}
	}
	current_stage = 1

	while current_stage <= len(stages):
		for algorithm_class, algorithm_name in algorithms:
			print(f"Running {algorithm_name} for Stage {current_stage}...")
			# Update config based on the current stage
			stage_config = config.copy()
			if current_stage == 1:
				stage_config["traffic_density"] = 0.1
				stage_config["horizon"] = 500
				stage_config["map"] = np.random.choice(
					["S", "C", "r"])  # Simple maps with straight, circular, and ramp sections
			elif current_stage == 2:
				stage_config["traffic_density"] = 0.3
				stage_config["horizon"] = 1000
				stage_config["map"] = np.random.choice(
					["X", "Y", "Z"])  # Intermediate maps with intersections and splits
			elif current_stage == 3:
				stage_config["traffic_density"] = 0.5
				stage_config["horizon"] = 2000
				stage_config["map"] = np.random.choice(
					["O", "C", "Y", "S"])  # Complex maps with roundabouts, curves, and more

			collector = DataCollector(stage_config)
			# Use train_and_evaluate_baseline for training and evaluation
			model = None  # Initialize model as None to pass on the first run
			while True:
				metrics, model = train_and_evaluate_baseline(algorithm_class, algorithm_name, collector, stage_config,
															 model, total_timesteps=int(config["training"]["steps"]))
				metrics["Stage"] = current_stage
				results.append(metrics)

				# Evaluate to determine if the agent should progress to the next stage
				criteria = stage_criteria[current_stage]
				progress = False
				success_rate = metrics.get("Success Rate", 0)
				collision_rate = metrics.get("Collision Rate per Episode", 1.0)

				if "success_rate" in criteria and success_rate >= criteria["success_rate"]:
					progress = True
				elif "collision_rate" in criteria and collision_rate <= criteria["collision_rate"]:
					progress = True

				if progress:
					print(f"{algorithm_name} met criteria for Stage {current_stage}, progressing to next stage...")
					# Keep the model to continue training in the next stage, supporting curriculum learning
					current_stage += 1
					break
				else:
					print(f"{algorithm_name} did not meet criteria for Stage {current_stage}, repeating stage...")
					# Continue with the current model for another round of training at the same stage
					continue

	# Save results to DataFrame for future comparison
	df = pd.DataFrame(results)
	df.to_csv("curriculum_baseline_evaluation_results.csv", index=False)
	display_results(df)


# Function to display and compare results in a visually appealing way
def display_results(df):
	# Set up the plotting style
	sns.set(style="whitegrid")

	# Plot Success Rate for each algorithm at each stage
	plt.figure(figsize=(12, 6))
	sns.barplot(x="Stage", y="Success Rate", hue="Algorithm", data=df)
	plt.title("Success Rate Comparison by Algorithm and Stage")
	plt.ylabel("Success Rate")
	plt.xlabel("Stage")
	plt.show()

	# Plot Average Reward per Episode
	plt.figure(figsize=(12, 6))
	sns.barplot(x="Stage", y="Average Reward per Episode", hue="Algorithm", data=df)
	plt.title("Average Reward per Episode by Algorithm and Stage")
	plt.ylabel("Average Reward")
	plt.xlabel("Stage")
	plt.show()

	# Plot Collision Rate per Episode
	plt.figure(figsize=(12, 6))
	sns.barplot(x="Stage", y="Collision Rate per Episode", hue="Algorithm", data=df)
	plt.title("Collision Rate per Episode by Algorithm and Stage")
	plt.ylabel("Collision Rate")
	plt.xlabel("Stage")
	plt.show()

	# Plot Average Distance Traveled
	plt.figure(figsize=(12, 6))
	sns.barplot(x="Stage", y="Average Distance Traveled", hue="Algorithm", data=df)
	plt.title("Average Distance Traveled by Algorithm and Stage")
	plt.ylabel("Distance Traveled")
	plt.xlabel("Stage")
	plt.show()


# Example usage
if __name__ == "__main__":
	run_all_baselines_with_curriculum()
