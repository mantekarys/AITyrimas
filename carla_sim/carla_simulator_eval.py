import yaml
from stable_baselines3 import PPO
import torch
import carla
from carla_env2 import CustomCarlaEnv
import utils

def test_policy(
    policy_file: str,
    frames_count: int = 1000,
    config: str = "main.yaml",
    just_embeddings=False,
) -> None:
    test_config = yaml.safe_load(open(f"configs/{config}", "r"))
    
    env = CustomCarlaEnv()

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
