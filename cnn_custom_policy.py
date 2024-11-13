import math
from typing import Any, Callable, Dict, Tuple, Union

import torch
import torch.nn as nn
from torchvision import models, transforms
import gymnasium as gym
from stable_baselines3.common.policies import ActorCriticPolicy
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor


class CustomResNetExtractor(BaseFeaturesExtractor):
    def __init__(self, observation_space: gym.spaces.Dict):
        super().__init__(observation_space, features_dim=2048)

        # Load pretrained ResNet50 model
        resnet = models.resnet50(weights = "ResNet50_Weights.DEFAULT")
        self.resnet = nn.Sequential(*list(resnet.children())[:-1])  # Remove the classifier

        # Normalization transform
        self.normalize = transforms.Normalize(
            mean=[0.485, 0.456, 0.406],  # ImageNet mean
            std=[0.229, 0.224, 0.225]    # ImageNet std
        )

    def forward(self, observations: Dict[str, torch.Tensor]) -> torch.Tensor:
        images = observations["image"]  # Shape: [batch_size, H, W, C, F]
        # print(f"Original images shape: {images.shape}")

        if images.dim() != 5:
            raise ValueError(f"Expected images to have 5 dimensions, but got {images.dim()} dimensions.")

        batch_size, H, W, C, F = images.shape

        # Permute and reshape to bring frames into batch dimension
        images = images.permute(0, 4, 3, 1, 2)  # Now [batch_size, F, C, H, W]
        images = images.reshape(batch_size * F, C, H, W)  # Now [batch_size * F, C, H, W]
        # print(f"Images reshaped for model input: {images.shape}")
        # raise ValueError(f"STOP")
        
        # Convert to float and normalize
        images = images.float() / 255.0
        images = self.normalize(images)

        # Resize images if necessary
        if images.shape[2] != 224 or images.shape[3] != 224:
            images = nn.functional.interpolate(images, size=(224, 224), mode='bilinear', align_corners=False)

        # Pass images through ResNet
        features = self.resnet(images)  # Output: [batch_size * F, 2048, 1, 1]
        features = features.view(features.size(0), -1)  # Now [batch_size * F, 2048]

        # Reshape to [batch_size, F, feature_dim]
        features = features.view(batch_size, F, -1)

        # Aggregate features over frames
        features = features.mean(dim=1)  # Now [batch_size, 2048]

        return features




class CustomNetwork(nn.Module):
    """
    Custom network for policy and value function.
    It receives as input the features extracted by the features extractor.

    :param feature_dim: dimension of the features extracted with the features_extractor (e.g. features from a CNN)
    :param last_layer_dim_pi: (int) number of units for the last layer of the policy network
    :param last_layer_dim_vf: (int) number of units for the last layer of the value network
    """

    def __init__(
        self,
        feature_dim: int,
        last_layer_dim_pi: int = 64,
        last_layer_dim_vf: int = 64,
    ):
        super().__init__()

        # IMPORTANT:
        # Save output dimensions, used to create the distributions
        self.latent_dim_pi = last_layer_dim_pi
        self.latent_dim_vf = last_layer_dim_vf

        # Policy network
        self.policy_net = nn.Sequential(
            nn.Linear(feature_dim, 256),
            nn.LayerNorm((256)),
            nn.ReLU(),
            nn.Linear(256, 64),
            nn.LayerNorm(64),
            nn.ReLU(),
        )
        # Value network
        self.value_net = nn.Sequential(
            nn.Linear(feature_dim, 256, bias=False),
            nn.LayerNorm((256)),
            nn.ReLU(),
            nn.Linear(256, 64, bias=False),
            nn.LayerNorm(64),
            nn.ReLU(),
        )

    def forward(self, features: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        :return: (th.Tensor, th.Tensor) latent_policy, latent_value of the specified network.
            If all layers are shared, then ``latent_policy == latent_value``
        """
        return self.forward_actor(features), self.forward_critic(features)

    def forward_actor(self, features: torch.Tensor) -> torch.Tensor:
        return self.policy_net(features)

    def forward_critic(self, features: torch.Tensor) -> torch.Tensor:
        return self.value_net(features)
    


class CustomResNetPolicy(ActorCriticPolicy):
    def __init__(
        self,
        observation_space: gym.spaces.Space,
        action_space: gym.spaces.Space,
        lr_schedule: Callable[[float], float],
        *args,
        **kwargs,
    ):
        super().__init__(
            observation_space,
            action_space,
            lr_schedule,
            ortho_init=False,
            features_extractor_class=CustomResNetExtractor,
            features_extractor_kwargs={},
            optimizer_class=torch.optim.AdamW,
            optimizer_kwargs={},
            *args,
            **kwargs,
        )

    def _build_mlp_extractor(self) -> None:
        # Using the same MLP network as before for policy and value extraction
        self.mlp_extractor = CustomNetwork(self.features_dim)
