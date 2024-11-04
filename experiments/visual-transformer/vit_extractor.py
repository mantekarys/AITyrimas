import timm
import torch

from stable_baselines3.common.policies import ActorCriticPolicy
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from stable_baselines3 import PPO
import gymnasium as gym
import gymnasium.spaces
import torch.nn as nn

class ViTFeatureExtractor(BaseFeaturesExtractor):
    # 224 * 224 * 3 * 4
    # image size 224 * 224, 3 channels, 4 in stack
    def __init__(self, observation_space: gym.spaces.Dict, features_dim: int = 1):
        super(ViTFeatureExtractor, self).__init__(observation_space, features_dim)
        print("Init ViTFeatureExtractor")
        print(observation_space)
        
        # assert that hey exists in observation_space
        assert "image" in observation_space.spaces, "Image observation is missing!"
        
        # Load a pre-trained Vision Transformer from timm
        self.vit = timm.create_model('vit_base_patch16_224', pretrained=True, num_classes=0)
        self.vit.eval()
    
        subspaces = observation_space.spaces["image"].shape[1] // 4 * observation_space.spaces["image"].shape[2] // 4
        print(subspaces)
        # Ensure the transformer outputs the correct number of features
        print(self.vit.num_features)
        self._features_dim = self.vit.num_features

    def forward(self, observations: torch.Tensor) -> torch.Tensor:
        print('forward')
        
        images: torch.Tensor = observations["image"]
        print(images.shape)
        
        test = self.vit(images)
        print (test)
        return self.vit(images)

# class ViTActorCriticPolicy(ActorCriticPolicy):
#     def __init__(self, *args, **kwargs):
#         super(ViTActorCriticPolicy, self).__init__(*args, **kwargs,
#             features_extractor_class=ViTFeatureExtractor,
#             features_extractor_kwargs=dict(features_dim=512))