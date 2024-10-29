import math
from typing import Any, Callable, Dict, Tuple, Union

import gymnasium as gym
import gymnasium.spaces
import numpy as np
import stable_baselines3 as sb3
import torch
import torch.nn as nn
import torch.optim
from gymnasium.spaces import Box
from stable_baselines3.common.policies import ActorCriticPolicy
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor


class CustomCombinedExtractor(BaseFeaturesExtractor):
    def __init__(self, observation_space: gym.spaces.Dict):
        # We do not know features-dim here before going over all the items,
        # so put something dummy for now. PyTorch requires calling
        # nn.Module.__init__ before adding modules
        super().__init__(observation_space, features_dim=1)
        output_vector_size = 0

        # torch.Size([2, 84, 84, 3, 4])
        # image shape: batch, {x, y}, chanels, time stack 4
        image_shape: Box = observation_space.spaces["image"]

        chanels = math.prod(image_shape.shape[-2:])
        self.net = nn.Sequential(
            nn.Conv2d(
                in_channels=chanels,
                out_channels=16,
                kernel_size=8,
                stride=4,
                bias=False,
            ),
            nn.LayerNorm([16, 20, 20]),
            nn.ReLU(),
            nn.Conv2d(
                in_channels=16, out_channels=32, kernel_size=4, stride=2, bias=False
            ),
            nn.LayerNorm([32, 9, 9]),
            nn.ReLU(),
            nn.Flatten(),
        )

        # Update the features dim manually
        self._features_dim = 2592
        # self._features_dim = output_vector_size

    def forward(self, observations) -> torch.Tensor:
        images: torch.Tensor = observations["image"]
        shape = images.shape
        stacked_chanels = images.reshape(
            shape=tuple([*shape[:-2], shape[-2] * shape[-1]])
        )
        chanel_second = stacked_chanels.permute((0, 3, 1, 2))
        # input: n,c,h,v
        res = self.net(chanel_second)

        return res
        # Return a (B, self._features_dim) PyTorch tensor, where B is batch dimension.
        return torch.cat(encoded_tensor_list, dim=1)


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


class CustomCNNPolicy(ActorCriticPolicy):
    def __init__(
        self,
        observation_space: gymnasium.spaces.Space,
        action_space: gymnasium.spaces.Space,
        lr_schedule: Callable[[float], float],
        *args,
        **kwargs,
    ):
        super().__init__(
            observation_space,
            action_space,
            lr_schedule,
            ortho_init=False,
            features_extractor_class=CustomCombinedExtractor,
            features_extractor_kwargs={},
            optimizer_class=torch.optim.AdamW,
            optimizer_kwargs={},
            *args,
            **kwargs,
        )

    def _build_mlp_extractor(self) -> None:
        self.mlp_extractor = CustomNetwork(self.features_dim)
