# DQN - Deep Q-Network Implementation

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/caochengrui/DQN/blob/main/DQN.ipynb)
[![Python 3.8+](https://img.shields.io/badge/python-3.8%2B-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-%3E%3D2.4.0-ee4c2c.svg)](https://pytorch.org/)

A modular implementation of the Deep Q-Network (DQN) algorithm built with [Gymnasium](https://gymnasium.farama.org/) and [PyTorch](https://pytorch.org/).

This repository now supports both common DQN input setups:

- vector observations with an MLP-based `QNetwork`
- pixel observations with a Nature-DQN-style `CNNQNetwork`

It also includes a replay buffer, epsilon-greedy data collection helpers, evaluation utilities, video recording helpers, visual observation wrappers, and a notebook with end-to-end training examples.

---

## Table of Contents

- [Overview](#overview)
- [Algorithm](#algorithm)
- [Features](#features)
- [Project Structure](#project-structure)
- [Installation](#installation)
- [Dependencies](#dependencies)
- [Public API](#public-api)
- [Quick Start](#quick-start)
- [Visual Observation Pipeline](#visual-observation-pipeline)
- [Evaluation and Video Recording](#evaluation-and-video-recording)
- [Supported Environments and Scope](#supported-environments-and-scope)
- [Acknowledgements](#acknowledgements)

---

## Overview

The codebase focuses on reusable DQN building blocks rather than a large training framework. It implements the standard DQN ingredients:

- an online Q-network
- a target network
- an experience replay buffer
- epsilon-greedy exploration
- helper utilities for evaluation and video recording

Two network backbones are provided out of the box:

- `QNetwork`: a 2-hidden-layer MLP for 1D vector observations
- `CNNQNetwork`: a convolutional Q-network for image observations, grayscale inputs, and frame-stacked inputs

The repository is intentionally lightweight. It provides an installable package plus [DQN.ipynb](DQN.ipynb) for training experiments, but it does not ship a standalone CLI trainer or advanced variants such as Double DQN, Dueling DQN, or prioritized replay.

---

## Algorithm

For a non-terminal transition, the standard DQN target is:

$$
y_t = r_t + \gamma (1 - d_t) \max_{a'} Q_{\text{target}}(s_{t+1}, a')
$$

where:

- $r_t$ is the reward
- $\gamma$ is the discount factor
- $d_t$ indicates whether the episode terminated
- $Q_{\text{target}}$ is the target network

A typical training loop in this repository looks like:

1. collect transitions with an epsilon-greedy policy
2. store them in the replay buffer
3. sample a mini-batch
4. compute TD targets with the target network
5. update the online Q-network
6. periodically synchronize the target network
7. evaluate the policy

Implementation note: the replay buffer stores `terminated` flags, while time-limit truncations are handled by resetting the environment during collection instead of being treated as absorbing terminal states.

---

## Features

- MLP-based DQN for vector observations via `QNetwork`
- CNN-based DQN for image observations via `CNNQNetwork`
- replay buffer with NumPy storage and `.to_torch()` conversion
- epsilon-greedy action selection and linear exploration schedule
- one-step collection helper for notebook-style training loops
- visual preprocessing wrappers for pixel observations
- optional video recording during evaluation
- Google Colab notebook for interactive experimentation
- installable package via `pip`

---

## Project Structure

```text
DQN/
├── DQN/                        # Core package
│   ├── __init__.py             # Public package exports
│   ├── collect_data.py         # Epsilon-greedy action selection and rollout helpers
│   ├── evaluation.py           # Evaluation and optional video recording
│   ├── q_network.py            # MLP and CNN Q-network definitions
│   ├── replay_buffer.py        # Replay buffer and batch containers
│   └── wrappers.py             # Pixel preprocessing wrappers and env factory
├── DQN.ipynb                   # End-to-end notebook examples
├── custom_utils.py             # Notebook helper for displaying recorded videos
├── pyproject.toml              # Packaging, dependencies, and tool configuration
└── README.md                   # Project documentation
```

---

## Installation

### Option 1: Install directly from GitHub

```bash
pip install "git+https://github.com/caochengrui/DQN.git"
```

### Option 2: Clone and install locally

```bash
git clone https://github.com/caochengrui/DQN.git
cd DQN
pip install -e .
```

### Option 3: Open the notebook in Google Colab

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/caochengrui/DQN/blob/main/DQN.ipynb)

### Optional setup

If you want to record evaluation videos, install `ffmpeg` first.

On Debian/Ubuntu or Google Colab:

```bash
sudo apt-get update
sudo apt-get install -y ffmpeg
```

On macOS:

```bash
brew install ffmpeg
```

If you want to run `FlappyBird-v0`, install the environment separately:

```bash
pip install "flappy-bird-gymnasium @ git+https://github.com/araffin/flappy-bird-gymnasium@patch-1"
```

---

## Dependencies

Core runtime dependencies from `pyproject.toml`:

- Python >= 3.8
- PyTorch >= 2.4.0
- Gymnasium >= 0.29.1, < 1.1.0 with `classic-control` and `other` extras
- NumPy
- scikit-learn
- opencv-python >= 4.6.0

The repository also keeps tool configuration for `ruff`, `black`, and `mypy` in `pyproject.toml`.

---

## Public API

The package root exports the most common components:

- `QNetwork`: MLP for discrete-action environments with 1D `Box` observations
- `CNNQNetwork`: CNN for image observations, grayscale frames, or frame stacks
- `ReplayBuffer`: ring-buffer replay storage
- `epsilon_greedy_action_selection`: action selection helper
- `collect_one_step`: collect one transition and store it in the replay buffer
- `linear_schedule`: linear epsilon schedule
- `PixelObservationWrapper`
- `GrayscaleWrapper`
- `ResizeWrapper`
- `FrameStack`
- `MaxAndSkipEnv`
- `make_visual_env`: convenience factory for image-based observations

Additional helpers are available from submodules:

- `DQN.evaluation.evaluate_policy`
- `custom_utils.notebook_show_videos`

---

## Quick Start

### Vector observations

```python
import gymnasium as gym
import torch as th
import torch.nn.functional as F
from torch import optim

from DQN import QNetwork, ReplayBuffer, collect_one_step, linear_schedule

env = gym.make("CartPole-v1")
obs, _ = env.reset()

q_net = QNetwork(env.observation_space, env.action_space)
target_net = QNetwork(env.observation_space, env.action_space)
target_net.load_state_dict(q_net.state_dict())

optimizer = optim.Adam(q_net.parameters(), lr=1e-3)
replay_buffer = ReplayBuffer(
    buffer_size=100_000,
    observation_space=env.observation_space,
    action_space=env.action_space,
)

for step in range(20_000):
    epsilon = linear_schedule(1.0, 0.05, step, 10_000)
    obs = collect_one_step(env, q_net, replay_buffer, obs, exploration_rate=epsilon)

    if not replay_buffer.is_full and replay_buffer.current_idx < 32:
        continue

    batch = replay_buffer.sample(32).to_torch()

    with th.no_grad():
        next_q_values = target_net(batch.next_observations).max(dim=1).values
        td_target = batch.rewards + 0.99 * next_q_values * (~batch.terminateds)

    current_q_values = q_net(batch.observations).gather(1, batch.actions).squeeze(1)
    loss = F.mse_loss(current_q_values, td_target)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    if step % 500 == 0:
        target_net.load_state_dict(q_net.state_dict())
```

### Pixel observations

```python
from DQN import CNNQNetwork, ReplayBuffer, make_visual_env

env = make_visual_env(
    "CartPole-v1",
    frame_stack=4,
    resize_shape=(84, 84),
    grayscale=True,
)

q_net = CNNQNetwork(env.observation_space, env.action_space)
replay_buffer = ReplayBuffer(
    buffer_size=50_000,
    observation_space=env.observation_space,
    action_space=env.action_space,
)
```

`CNNQNetwork` automatically handles common image layouts such as `(H, W)`, `(H, W, C)`, `(C, H, W)`, and frame-stacked inputs such as `(4, 84, 84)`. For `uint8` observations it also normalizes inputs to `[0, 1]`.

For a complete training workflow, evaluation loop, and notebook-friendly examples, see [DQN.ipynb](DQN.ipynb).

---

## Visual Observation Pipeline

`make_visual_env()` builds a preprocessing pipeline for image-based DQN experiments.

The wrappers are applied in this order:

1. `PixelObservationWrapper` converts the environment output to rendered RGB frames
2. `MaxAndSkipEnv` optionally applies frame skipping with max pooling over the last 2 frames
3. `GrayscaleWrapper` optionally converts RGB frames to grayscale
4. `ResizeWrapper` resizes frames to the target resolution
5. `FrameStack` stacks consecutive frames along the first axis
6. `gym.wrappers.RecordEpisodeStatistics` optionally records episode returns and lengths

Typical usage:

- use the default `use_pixel_wrapper=True` for environments such as `CartPole-v1` that normally return vector observations
- set `use_pixel_wrapper=False` for environments that already emit image observations
- set `frame_skip > 0` when you want Atari-style action repeat and max pooling

---

## Evaluation and Video Recording

Evaluation helpers live in `DQN.evaluation`.

```python
from DQN import make_visual_env
from DQN.evaluation import evaluate_policy
from custom_utils import notebook_show_videos

eval_env = make_visual_env("CartPole-v1", render_mode="rgb_array")
evaluate_policy(eval_env, q_net, n_eval_episodes=5, video_name="cartpole_demo")

notebook_show_videos("logs/videos", prefix="cartpole_demo")
```

When `video_name` is provided and the environment uses `render_mode="rgb_array"`, videos are saved under `logs/videos/`.

---

## Supported Environments and Scope

This repository is designed for discrete-action Gymnasium environments.

Supported out of the box:

- vector-observation environments with 1D `spaces.Box` observations, such as `CartPole-v1`
- pixel-observation environments using `CNNQNetwork`
- environments converted to pixels with `make_visual_env()`
- `FlappyBird-v0` after installing the optional environment package

Important limitations:

- the action space must be `spaces.Discrete`
- `QNetwork` expects 1D vector observations
- `CNNQNetwork` expects 2D or 3D image observations
- there is no built-in support for continuous control
- advanced DQN variants are not included in the package API

If you want to work with Atari-like image environments, you can already use `CNNQNetwork` and the wrappers in this repository instead of replacing the network from scratch.

---

## Acknowledgements

- This project is based on the [RLSS 2023 DQN tutorial](https://github.com/araffin/rlss23-dqn) by [Antonin Raffin](https://github.com/araffin).
- Some hyperparameter choices are inspired by the [RL Baselines3 Zoo](https://github.com/DLR-RM/rl-baselines3-zoo).
- The optional Flappy Bird environment uses the [patch-1 fork of flappy-bird-gymnasium](https://github.com/araffin/flappy-bird-gymnasium/tree/patch-1).
