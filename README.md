# DQN — Deep Q-Network Implementation

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/caochengrui/DQN/blob/main/DQN.ipynb)
[![Python 3.8+](https://img.shields.io/badge/python-3.8%2B-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-%3E%3D2.4.0-ee4c2c.svg)](https://pytorch.org/)

A clean, modular implementation of the **Deep Q-Network (DQN)** algorithm built with [Gymnasium](https://gymnasium.farama.org/) and [PyTorch](https://pytorch.org/). The repository contains reusable DQN components, an interactive training notebook, and utilities for evaluation and video recording.

---

## Table of Contents

- [Overview](#overview)
- [Project Structure](#project-structure)
- [Algorithm](#algorithm)
- [Features](#features)
- [Installation](#installation)
- [Quick Start](#quick-start)
- [Supported Environments](#supported-environments)
- [Dependencies](#dependencies)
- [Acknowledgements](#acknowledgements)

---

## Overview

This project implements the core ideas of DQN from [Mnih et al. (2015)](https://www.nature.com/articles/nature14236), including:

- **Q-Network**: a fully connected neural network that predicts Q-values for each discrete action.
- **Target Network**: a delayed copy of the online network used to compute more stable TD targets.
- **Experience Replay Buffer**: a replay buffer for storing transitions and sampling mini-batches.
- **Epsilon-Greedy Exploration**: a linear exploration schedule during training.
- **Evaluation Utilities**: helper functions for evaluation and optional video recording.

---

## Project Structure

```text
DQN/
├── DQN/                        # Core DQN package
│   ├── __init__.py             # Package exports
│   ├── collect_data.py         # Data collection and exploration utilities
│   ├── evaluation.py           # Policy evaluation and video recording helpers
│   ├── q_network.py            # Q-network architecture (2-hidden-layer MLP)
│   └── replay_buffer.py        # Experience replay buffer
├── DQN.ipynb                   # Notebook with training and experimentation code
├── custom_utils.py             # Notebook video display utilities
├── pyproject.toml              # Project metadata and dependencies
└── README.md                   # Project documentation
```

---

## Algorithm

For a non-terminal transition, the DQN target is:

$$
y_t = r_t + \gamma (1 - d_t) \max_{a'} Q_{\text{target}}(s_{t+1}, a')
$$

where:

- $r_t$ is the reward,
- $\gamma$ is the discount factor,
- $d_t$ indicates whether the episode terminated,
- $Q_{\text{target}}$ is the target network.

The training loop follows the standard DQN pipeline:

1. Collect transitions with an epsilon-greedy policy.
2. Store transitions in the replay buffer.
3. Sample a mini-batch from the buffer.
4. Compute TD targets with the target network.
5. Update the online Q-network.
6. Periodically sync the target network.
7. Evaluate the policy.

---

## Features

- **Modular implementation** of the main DQN building blocks
- **Target network** support for more stable learning
- **Replay buffer** for off-policy training
- **Linear epsilon schedule** for exploration
- **Evaluation utilities** with optional video recording
- **Google Colab notebook** for interactive experiments
- **Installable package** via `pip`

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

### Optional dependencies

If you want to record evaluation videos, install **ffmpeg** first.

On Debian/Ubuntu or Google Colab:

```bash
sudo apt-get update
sudo apt-get install -y ffmpeg
```

On macOS:

```bash
brew install ffmpeg
```

If you want to run **FlappyBird-v0**, install the environment separately:

```bash
pip install flappy-bird-gymnasium
```

---

## Quick Start

```python
import gymnasium as gym
import torch as th
from torch import optim

from DQN import QNetwork, ReplayBuffer

# Create environment
env = gym.make("CartPole-v1")

# Create networks
q_net = QNetwork(env.observation_space, env.action_space)
target_net = QNetwork(env.observation_space, env.action_space)
target_net.load_state_dict(q_net.state_dict())

# Optimizer and replay buffer
optimizer = optim.Adam(q_net.parameters(), lr=1e-3)
replay_buffer = ReplayBuffer(
    buffer_size=100_000,
    observation_space=env.observation_space,
    action_space=env.action_space,
)
```

A minimal DQN update looks like this:

```python
def dqn_update(q_net, target_net, optimizer, replay_buffer, batch_size, gamma):
    replay_data = replay_buffer.sample(batch_size).to_torch()

    with th.no_grad():
        next_q_values = target_net(replay_data.next_observations)
        next_q_values, _ = next_q_values.max(dim=1)
        should_bootstrap = th.logical_not(replay_data.terminateds)
        td_target = replay_data.rewards + gamma * next_q_values * should_bootstrap

    q_values = q_net(replay_data.observations)
    current_q_values = th.gather(q_values, dim=1, index=replay_data.actions).squeeze(dim=1)

    loss = ((current_q_values - td_target) ** 2).mean()
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    return loss.item()
```

For a complete training loop, evaluation, and experimentation code, see [DQN.ipynb](DQN.ipynb).

---

## Supported Environments

This repository currently includes code that is suitable for **discrete-action environments** such as:

- **CartPole-v1**
- **FlappyBird-v0** (with an additional environment package installed)

The implementation is designed around:

- a **1D observation vector**,
- a **discrete action space**, and
- an **MLP-based Q-network**.

If you want to apply it to image-based environments such as Atari, you would typically replace the MLP with a CNN-based Q-network.

---

## Dependencies

Core dependencies defined in `pyproject.toml`:

- Python >= 3.8
- PyTorch >= 2.4.0
- Gymnasium >= 0.29.1, < 1.1.0
- NumPy
- scikit-learn

---

## Acknowledgements

- This project is based on the [RLSS 2023 DQN tutorial](https://github.com/araffin/rlss23-dqn) by [Antonin Raffin](https://github.com/araffin).
- Some hyperparameter choices are inspired by the [RL Baselines3 Zoo](https://github.com/DLR-RM/rl-baselines3-zoo).
- The optional Flappy Bird environment can be installed from [flappy-bird-gymnasium](https://github.com/markub3327/flappy-bird-gymnasium).

