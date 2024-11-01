"""
main.py

This module contains the main script that trains and tests the agent.
"""

from environments.visual_gym_environment import VisualGymEnvironment
from agents.dqn_agent import DQNAgent
from trainers.trainer import Trainer
import torch
import numpy as np

# Set the random seed
seed = 42
np.random.seed(seed)
torch.manual_seed(seed)

# Create the environment
env = VisualGymEnvironment("CartPole-v1")

# Create the agent
agent = DQNAgent(512, 2)

# Train the agent
trainer = Trainer(agent, env)
trainer.train(10000)

# Test the agent
tester = Trainer(agent, env)
results = tester.test(10)
print(results)
