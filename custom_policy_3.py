import math
from typing import Any, Callable, Dict, Tuple, Union

import dinov2
import gymnasium as gym
import gymnasium.spaces
import numpy as np
import stable_baselines3 as sb3
import torch
import torch.nn as nn
import torch.optim
from dinov2.hub.backbones import dinov2_vitb14_reg
from gymnasium.spaces import Box
from stable_baselines3.common.policies import ActorCriticPolicy
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor


class CustomCombinedExtractor(BaseFeaturesExtractor):
    def __init__(self, observation_space: gym.spaces.Dict):
        # We do not know features-dim here before going over all the items,
        # so put something dummy for now. PyTorch requires calling
        # nn.Module.__init__ before adding modules
        super().__init__(observation_space, features_dim=1)
        # obs["vit_embeddings"] shape: batch, time stack 4, 256, 384

        self.embedding_compression_1 = nn.Sequential(
            nn.Conv2d(
                in_channels=384, out_channels=64, kernel_size=1, stride=1, bias=False
            ),
            nn.LayerNorm((64, 16, 16)),
            nn.GELU(),
        )
        self.compression_2_and_linear = nn.Sequential(
            nn.Conv2d(
                in_channels=64 * 4, out_channels=32, kernel_size=5, stride=1, bias=False
            ),
            nn.LayerNorm((32, 12, 12)),
            nn.GELU(),
            nn.Conv2d(
                in_channels=32, out_channels=8, kernel_size=3, stride=1, bias=False
            ),
            nn.LayerNorm((8, 10, 10)),
            nn.GELU(),
            
            nn.Flatten(),
            nn.Linear(in_features=8*100, out_features=128, bias=False),
            nn.LayerNorm((128)),
            nn.GELU(),
        )

        self._features_dim = 128

    def forward(self, observations) -> torch.Tensor:
        embeddings: torch.Tensor = observations["vit_embeddings"]
        shape = embeddings.shape
        stacked_frames_count = shape[0] * shape[1]
        image_width = int(shape[2] ** (1 / 2))

        chanels_third = embeddings.permute((0, 1, 3, 2))
        stacked_frames = chanels_third.reshape(
            (stacked_frames_count, shape[3], image_width, image_width)
        )

        res = self.embedding_compression_1(stacked_frames)
        res = res.reshape((shape[0], shape[1] * 64, 16, 16))
        res = self.compression_2_and_linear(res)
        return res


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
            nn.Linear(feature_dim, 64, bias=False),
            nn.LayerNorm(64),
            nn.GELU(),
        )
        # Value network
        self.value_net = nn.Sequential(
            nn.Linear(feature_dim, 64, bias=False),
            nn.LayerNorm(64),
            nn.GELU(),
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


class CustomViTPolicy2(ActorCriticPolicy):
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
        parameters_to_train = []
        for module_name, module in self.named_children():
            if module_name == "features_extractor":
                extractor_parts = dict(module.named_children())
                parameters_to_train += extractor_parts[
                    "embedding_compression_1"
                ].parameters()
                parameters_to_train += extractor_parts[
                    "compression_2_and_linear"
                ].parameters()
            else:
                parameters_to_train += module.parameters()
        self.optimizer = torch.optim.AdamW(
            params=parameters_to_train, lr=lr_schedule(1), **self.optimizer_kwargs
        )

    def _build_mlp_extractor(self) -> None:
        self.mlp_extractor = CustomNetwork(self.features_dim)
