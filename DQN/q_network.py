from typing import Optional, Type

import numpy as np
import torch as th
import torch.nn as nn
from gymnasium import spaces


class QNetwork(nn.Module):
    """
    A Q-Network for the DQN algorithm
    to estimate the q-value for a given observation.

    :param observation_space: Observation space of the env,
        contains information about the observation type and shape.
    :param action_space: Action space of the env,
        contains information about the number of actions.
    :param n_hidden_units: Number of units for each hidden layer.
    :param activation_fn: Activation function (ReLU by default)
    """

    def __init__(
        self,
        observation_space: spaces.Box,
        action_space: spaces.Discrete,
        n_hidden_units: int = 64,
        activation_fn: Type[nn.Module] = nn.ReLU,
    ) -> None:
        super().__init__()
        # Assume 1d space
        obs_dim = observation_space.shape[0]
        # Retrieve the number of discrete actions
        n_actions = int(action_space.n)
        # Create the q network (2 fully connected hidden layers)
        self.q_net = nn.Sequential(
            nn.Linear(obs_dim, n_hidden_units),
            activation_fn(),
            nn.Linear(n_hidden_units, n_hidden_units),
            activation_fn(),
            nn.Linear(n_hidden_units, n_actions),
        )

    def forward(self, observations: th.Tensor) -> th.Tensor:
        """
        :param observations: A batch of observation (batch_size, obs_dim)
        :return: The Q-values for the given observations
            for all the action (batch_size, n_actions)
        """
        return self.q_net(observations)


class CNNQNetwork(nn.Module):
    """
    A CNN-based Q-Network for processing image observations,
    following the Nature DQN architecture (Mnih et al., 2015).

    Supports:
    - Image observations in (C, H, W) or (H, W, C) format (auto-detected).
    - Grayscale images of shape (H, W).
    - Automatic uint8 [0, 255] -> float32 [0, 1] normalization.
    - Frame-stacked observations, e.g. (4, 84, 84).

    :param observation_space: Observation space of the env (must be >= 2D image).
    :param action_space: Action space of the env (must be Discrete).
    :param n_hidden_units: Number of units in the fully connected layer after CNN.
    :param activation_fn: Activation function (ReLU by default).
    :param channels_last: If True, observations are (H, W, C). If False, (C, H, W).
        If None (default), auto-detected from observation_space shape.
    """

    def __init__(
        self,
        observation_space: spaces.Box,
        action_space: spaces.Discrete,
        n_hidden_units: int = 512,
        activation_fn: Type[nn.Module] = nn.ReLU,
        channels_last: Optional[bool] = None,
    ) -> None:
        super().__init__()
        obs_shape = observation_space.shape
        assert len(obs_shape) >= 2, (
            f"CNNQNetwork requires image observations (at least 2D), got shape {obs_shape}"
        )

        n_actions = int(action_space.n)
        self._is_uint8 = observation_space.dtype == np.uint8

        # Determine number of input channels and spatial dimensions
        if len(obs_shape) == 2:
            # Grayscale image (H, W) — no channel dimension
            self._channels_last = False
            self._is_2d = True
            n_channels = 1
            h, w = obs_shape
        elif len(obs_shape) == 3:
            self._is_2d = False
            if channels_last is not None:
                self._channels_last = channels_last
            else:
                # Auto-detect: if last dimension is small (<= 4), assume (H, W, C)
                # This covers RGB (3), RGBA (4), grayscale with keepdim (1)
                # Otherwise assume (C, H, W) — e.g. frame-stacked (4, 84, 84)
                self._channels_last = obs_shape[2] <= 4 and obs_shape[0] > 4

            if self._channels_last:
                h, w, n_channels = obs_shape
            else:
                n_channels, h, w = obs_shape
        else:
            raise ValueError(f"Unexpected observation shape: {obs_shape}")

        # Nature DQN CNN architecture
        self.cnn = nn.Sequential(
            nn.Conv2d(n_channels, 32, kernel_size=8, stride=4, padding=0),
            activation_fn(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=0),
            activation_fn(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=0),
            activation_fn(),
            nn.Flatten(),
        )

        # Compute the output size of the CNN by doing a forward pass with dummy data
        with th.no_grad():
            dummy = th.zeros(1, n_channels, h, w, dtype=th.float32)
            cnn_output_dim = self.cnn(dummy).shape[1]

        # Fully connected head: one hidden layer + output layer
        self.q_head = nn.Sequential(
            nn.Linear(cnn_output_dim, n_hidden_units),
            activation_fn(),
            nn.Linear(n_hidden_units, n_actions),
        )

    def forward(self, observations: th.Tensor) -> th.Tensor:
        """
        :param observations: A batch of image observations.
            Shape: (batch, C, H, W), (batch, H, W, C), or (batch, H, W).
        :return: Q-values for all actions, shape (batch, n_actions).
        """
        # Normalize uint8 observations from [0, 255] to [0, 1]
        if self._is_uint8:
            observations = observations.float() / 255.0
        else:
            observations = observations.float()

        # Handle channel dimensions
        if self._is_2d and observations.dim() == 3:
            # (batch, H, W) -> (batch, 1, H, W)
            observations = observations.unsqueeze(1)
        elif self._channels_last and observations.dim() == 4:
            # (batch, H, W, C) -> (batch, C, H, W)
            observations = observations.permute(0, 3, 1, 2)

        features = self.cnn(observations)
        return self.q_head(features)
