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
        super().__init__(observation_space, features_dim=512)

    def forward(self, observations: Dict[str, torch.Tensor]) -> torch.Tensor:
        # Get the pre-extracted ResNet features
        resnet_features = observations["resnet_features"]
        return resnet_features


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
