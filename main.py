"""
main.py

This module contains the main script that trains and tests the agent.

Ideally, the main script should be kept as simple as possible. It should
generally only create objects and run functions. Try not to place any complex
logic here and instead abstract it away.
"""

from environments import GymEnvironment, VisualGymEnvironment
from agents import PPOAgent
from trainers import Trainer
from vision import FeatureExtractor
from utils import DEVICE
import torch
import numpy as np

# Set the random seed
seed = 42
np.random.seed(seed)
torch.manual_seed(seed)

# Create the environment
env = VisualGymEnvironment("CartPole-v1", greyscale=True)
render_env = VisualGymEnvironment("CartPole-v1", render=True, greyscale=True)

# Instantiate the feature extractor
feature_extractor = FeatureExtractor(
    input_shape=env.observation_size,
    output_size=32,
    conv_channels=[2, 4],
    kernel_sizes=[3, 3],
    strides=[1, 1],
    paddings=[1, 1],
    pool_kernel_sizes=[4, 4],
    pool_strides=[4, 4]
).to(DEVICE)

# Create the agent
agent = PPOAgent(env.observation_size, env.action_size, feature_extractor=feature_extractor,
    normalize_advantages = False,
    n_steps = 4096,
    n_mini_batches = 4,
    n_training_steps = 10,
    gradient_clipping = 0.3
)

# Train the agent
trainer = Trainer(agent, env)
trainer.train(1000000)

# Test the agent
tester = Trainer(agent, render_env)
results = tester.test(10)
print(results)
