"""
Environment wrappers for visual (image-based) DQN.

Provides wrappers to convert any gymnasium environment into one that uses
pixel (image) observations, along with standard preprocessing:
grayscale conversion, resizing, and frame stacking.
"""

from collections import deque
from typing import Any, Optional, Tuple

import cv2
import gymnasium as gym
import numpy as np
from gymnasium import spaces


class PixelObservationWrapper(gym.ObservationWrapper):
    """
    Converts an environment's observations to rendered pixel images.

    The environment must be created with ``render_mode="rgb_array"`` so that
    ``env.render()`` returns an RGB numpy array.

    :param env: The environment to wrap. Must have ``render_mode="rgb_array"``.
    """

    def __init__(self, env: gym.Env) -> None:
        assert env.render_mode == "rgb_array", (
            f"PixelObservationWrapper requires render_mode='rgb_array', "
            f"got '{env.render_mode}'"
        )
        super().__init__(env)

        # Render a sample frame to determine the observation space
        env.reset()
        sample_frame: np.ndarray = env.render()  # type: ignore[assignment]
        assert sample_frame is not None, "env.render() returned None"

        self.observation_space = spaces.Box(
            low=0, high=255, shape=sample_frame.shape, dtype=np.uint8,
        )

    def observation(self, observation: np.ndarray) -> np.ndarray:
        """Replace the vector observation with the rendered pixel frame."""
        frame: np.ndarray = self.env.render()  # type: ignore[assignment]
        assert frame is not None, "env.render() returned None during observation"
        return frame


class GrayscaleWrapper(gym.ObservationWrapper):
    """
    Convert RGB observations to grayscale.

    :param env: The environment to wrap.
    :param keep_dim: If True, output shape is (H, W, 1); otherwise (H, W).
    """

    def __init__(self, env: gym.Env, keep_dim: bool = False) -> None:
        super().__init__(env)
        self.keep_dim = keep_dim

        old_space = env.observation_space
        assert isinstance(old_space, spaces.Box), "Observation space must be Box"
        assert len(old_space.shape) == 3 and old_space.shape[2] == 3, (
            f"Expected (H, W, 3) observations, got shape {old_space.shape}"
        )

        h, w = old_space.shape[0], old_space.shape[1]
        new_shape: Tuple[int, ...]
        if keep_dim:
            new_shape = (h, w, 1)
        else:
            new_shape = (h, w)

        self.observation_space = spaces.Box(
            low=0, high=255, shape=new_shape, dtype=np.uint8,
        )

    def observation(self, observation: np.ndarray) -> np.ndarray:
        """Convert an RGB observation to grayscale."""
        gray = cv2.cvtColor(observation, cv2.COLOR_RGB2GRAY)
        if self.keep_dim:
            gray = gray[:, :, np.newaxis]
        return gray


class ResizeWrapper(gym.ObservationWrapper):
    """
    Resize image observations to a target (height, width).

    :param env: The environment to wrap.
    :param shape: Target size as ``(height, width)``.
    """

    def __init__(self, env: gym.Env, shape: Tuple[int, int]) -> None:
        super().__init__(env)
        assert len(shape) == 2, f"shape must be (height, width), got {shape}"
        self.target_h, self.target_w = shape

        old_space = env.observation_space
        assert isinstance(old_space, spaces.Box), "Observation space must be Box"

        new_shape: Tuple[int, ...]
        if len(old_space.shape) == 2:
            new_shape = (self.target_h, self.target_w)
        elif len(old_space.shape) == 3:
            new_shape = (self.target_h, self.target_w, old_space.shape[2])
        else:
            raise ValueError(f"Unexpected observation shape: {old_space.shape}")

        self.observation_space = spaces.Box(
            low=0, high=255, shape=new_shape, dtype=np.uint8,
        )

    def observation(self, observation: np.ndarray) -> np.ndarray:
        """Resize the observation using area interpolation."""
        return cv2.resize(
            observation,
            (self.target_w, self.target_h),
            interpolation=cv2.INTER_AREA,
        )


class FrameStack(gym.Wrapper):
    """
    Stack the last ``num_stack`` frames along a new first axis.

    After wrapping, observations have shape ``(num_stack, *obs_shape)``.
    For example, grayscale 84x84 with 4-frame stack → ``(4, 84, 84)``,
    which is channels-first and ready for a CNN.

    Unlike gymnasium's ``FrameStackObservation`` (which may use ``LazyFrames``),
    this wrapper always returns plain numpy arrays, ensuring compatibility
    with the replay buffer.

    :param env: The environment to wrap.
    :param num_stack: Number of frames to stack.
    """

    def __init__(self, env: gym.Env, num_stack: int = 4) -> None:
        super().__init__(env)
        self.num_stack = num_stack
        self.frames: deque = deque(maxlen=num_stack)

        old_space = env.observation_space
        assert isinstance(old_space, spaces.Box), "Observation space must be Box"

        low = np.repeat(old_space.low[np.newaxis, ...], num_stack, axis=0)
        high = np.repeat(old_space.high[np.newaxis, ...], num_stack, axis=0)
        self.observation_space = spaces.Box(
            low=low, high=high, dtype=old_space.dtype,  # type: ignore[arg-type]
        )

    def _get_obs(self) -> np.ndarray:
        """Return stacked frames as a numpy array."""
        return np.array(self.frames)

    def reset(self, **kwargs):
        """Reset and fill the frame stack with the initial observation."""
        obs, info = self.env.reset(**kwargs)
        for _ in range(self.num_stack):
            self.frames.append(obs)
        return self._get_obs(), info

    def step(self, action):
        """Step the environment and push the new frame onto the stack."""
        obs, reward, terminated, truncated, info = self.env.step(action)
        self.frames.append(obs)
        return self._get_obs(), reward, terminated, truncated, info


class MaxAndSkipEnv(gym.Wrapper):
    """
    Skip every ``skip`` frames and return the max over the last 2 frames.

    This is a standard Atari preprocessing step to handle flickering sprites.

    :param env: The environment to wrap.
    :param skip: Number of frames to skip (default 4).
    """

    def __init__(self, env: gym.Env, skip: int = 4) -> None:
        super().__init__(env)
        self._skip = skip
        obs_space = env.observation_space
        assert isinstance(obs_space, spaces.Box)
        self._obs_buffer = np.zeros((2, *obs_space.shape), dtype=obs_space.dtype)

    def step(self, action):
        """Repeat action for ``skip`` frames; return max of last 2."""
        total_reward = 0.0
        terminated = truncated = False
        for i in range(self._skip):
            obs, reward, terminated, truncated, info = self.env.step(action)
            total_reward += float(reward)
            if i == self._skip - 2:
                self._obs_buffer[0] = obs
            if i == self._skip - 1:
                self._obs_buffer[1] = obs
            if terminated or truncated:
                break
        max_frame = self._obs_buffer.max(axis=0)
        return max_frame, total_reward, terminated, truncated, info


def make_visual_env(
    env_id: str,
    frame_stack: int = 4,
    resize_shape: tuple = (84, 84),
    grayscale: bool = True,
    frame_skip: int = 0,
    render_mode: str = "rgb_array",
    use_pixel_wrapper: bool = True,
    record_episode_stats: bool = True,
) -> gym.Env:
    """
    Create a gymnasium environment with visual (pixel) observations and
    standard image preprocessing.

    The preprocessing pipeline applied (in order):
    1. ``PixelObservationWrapper`` — use rendered pixels as observations
       (skipped if ``use_pixel_wrapper=False``, e.g. for Atari which already
       outputs images).
    2. ``MaxAndSkipEnv`` — frame skipping with max over last 2 frames
       (only if ``frame_skip > 0``).
    3. ``GrayscaleWrapper`` — convert RGB to grayscale.
    4. ``ResizeWrapper`` — resize to ``resize_shape``.
    5. ``FrameStack`` — stack ``frame_stack`` consecutive frames.
    6. ``RecordEpisodeStatistics`` — record episode returns & lengths.

    The final observation shape is ``(frame_stack, *resize_shape)`` for
    grayscale, or ``(frame_stack, H, W, C)`` for colour.

    :param env_id: Gymnasium environment ID.
    :param frame_stack: Number of frames to stack (default 4).
    :param resize_shape: Target ``(height, width)`` for resizing (default ``(84, 84)``).
    :param grayscale: Whether to convert observations to grayscale (default True).
    :param frame_skip: Number of frames to skip with max pooling (0 = disabled).
    :param render_mode: Render mode for the base environment.
    :param use_pixel_wrapper: Whether to apply ``PixelObservationWrapper``
        (set False for environments that already return images, e.g. Atari).
    :param record_episode_stats: Whether to wrap with ``RecordEpisodeStatistics``.
    :return: The wrapped environment.
    """
    env = gym.make(env_id, render_mode=render_mode)

    # 1. Convert vector observations to pixel observations
    if use_pixel_wrapper:
        env = PixelObservationWrapper(env)

    # 2. Frame skipping with max pooling
    if frame_skip > 0:
        env = MaxAndSkipEnv(env, skip=frame_skip)

    # 3. Grayscale
    if grayscale:
        env = GrayscaleWrapper(env)

    # 4. Resize
    if resize_shape is not None:
        env = ResizeWrapper(env, shape=resize_shape)

    # 5. Frame stacking
    if frame_stack > 1:
        env = FrameStack(env, num_stack=frame_stack)

    # 6. Record episode statistics
    if record_episode_stats:
        env = gym.wrappers.RecordEpisodeStatistics(env)

    return env
