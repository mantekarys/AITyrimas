import yaml
from stable_baselines3 import PPO
import torch
from carla_env2 import CustomCarlaEnv
import numpy as np
from torchvision import transforms
from typing import Tuple

def resize(image: np.ndarray, target_size: Tuple[int, int] = (224)):
    transformation = transforms.Resize(size=target_size)
    image: torch.Tensor = torch.tensor(image)
    shape = image.shape
    concatenated = image.reshape(shape=tuple([*shape[:-2], shape[-2] * shape[-1]]))
    chanel_second = concatenated.permute((0, 3, 1, 2))
    resized_image: torch.Tensor = transformation(chanel_second)
    chanel_last = resized_image.permute((0, 2, 3, 1))
    primary_shape = chanel_last.reshape(
        shape=tuple([shape[0], target_size[0], target_size[1], 3, 4])
    )
    return primary_shape

def test_policy(
    policy_file: str,
    frames_count: int = 1000,
    config: str = "main.yaml",
    just_embeddings=False,
) -> None:
    # test_config = yaml.safe_load(open(f"configs/{config}", "r"))
    
    env = CustomCarlaEnv()

    model = PPO.load(policy_file)

    try:
        obs = env.reset()
        # obs["image"] = resize(obs["image"], (224, 224))

        for _ in range(frames_count):
            with torch.no_grad():
                if just_embeddings:
                    obs = {"vit_embeddings": obs["vit_embeddings"]}
                action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = env.step(action)
            print("Reward: ", reward)
            print("-----")

            if terminated or truncated:
                obs, info = env.reset()
            # obs["image"] = resize(obs["image"], (224, 224))

        env.close()
    except Exception as e:
        env.close()
        raise e
    
if __name__ == "__main__":
    test_policy("../../models/chill-owl-867-1.zip", 2000, just_embeddings=True)
