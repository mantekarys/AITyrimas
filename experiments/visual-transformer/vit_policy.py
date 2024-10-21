import timm
import torch

from stable_baselines3.common.policies import ActorCriticPolicy
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from stable_baselines3 import PPO

from gym import spaces
import torch.nn as nn

class ViTFeatureExtractor(BaseFeaturesExtractor):
    def __init__(self, observation_space: spaces.Box, features_dim: int = 512):
        super(ViTFeatureExtractor, self).__init__(observation_space, features_dim)
        # Load a pre-trained Vision Transformer from timm
        self.vit = timm.create_model('vit_base_patch16_224', pretrained=True, num_classes=0)
        self.vit.eval()
        # Ensure the transformer outputs the correct number of features
        self.features_dim = self.vit.num_features

    def forward(self, observations: torch.Tensor) -> torch.Tensor:
        return self.vit(observations)

class ViTActorCriticPolicy(ActorCriticPolicy):
    def __init__(self, *args, **kwargs):
        super(ViTActorCriticPolicy, self).__init__(*args, **kwargs,
            features_extractor_class=ViTFeatureExtractor,
            features_extractor_kwargs=dict(features_dim=512))