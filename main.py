"""
main.py

This module contains the main script that trains and tests the agent.

Ideally, the main script should be kept as simple as possible. It should
generally only create objects and run functions. Try not to place any complex
logic here and instead abstract it away.
"""

from environments import GymEnvironment
from agents import PPOAgent
from trainers import Trainer
import torch
import numpy as np

# Set the random seed
seed = 42
np.random.seed(seed)
torch.manual_seed(seed)

# Create the environment
env = GymEnvironment("CartPole-v1")
render_env = GymEnvironment("CartPole-v1", render_mode="human")

# Create the agent
agent = PPOAgent(env.observation_size, env.action_size)

# Train the agent
trainer = Trainer(agent, env)
trainer.train(5000)

# Test the agent
tester = Trainer(agent, render_env)
results = tester.test(10)
print(results)
