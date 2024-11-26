import yaml
from stable_baselines3 import PPO
import torch
import carla
from carla_sim.carla_env import CarlaEnv
import utils

def test_policy(
    policy_file: str,
    frames_count: int = 1000,
    config: str = "main.yaml",
    just_embeddings=False,
) -> None:
    test_config = yaml.safe_load(open(f"configs/{config}", "r"))

    print("Connecting to CARLA server")
    client = carla.Client("4.210.242.233", 2000)
    client.set_timeout(60.0)
    
    print("Creating world")
    world = client.get_world()
    
    env = CarlaEnv(client, world, None)

    model = PPO.load(policy_file)

    obs = env.reset()
    print(obs.keys())
    obs["image"] = utils.resize(obs["image"], (224, 224))

    for _ in range(frames_count):
        with torch.no_grad():
            if just_embeddings:
                obs = {"vit_embeddings": obs["vit_embeddings"]}
            action, _ = model.predict(obs, deterministic=True)
        obs, reward, done, info = env.step(action)

        # obs["image"] = utils.resize(obs["image"], (224, 224))

    env.close()
    
if __name__ == "__main__":
    test_policy("models/chill-owl-867-1.zip", 2000, just_embeddings=True)
