"""
main.py

This module contains the main script that trains and tests the agent.
"""

from environments.gym_environment import GymEnvironment
from agents.dqn_agent import DQNAgent
from trainers.trainer import Trainer
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
agent = DQNAgent(4, 2)

# Train the agent
trainer = Trainer(agent, env)
trainer.train(10000)

# Test the agent
tester = Trainer(agent, render_env)
results = tester.test(10)
print(results)
