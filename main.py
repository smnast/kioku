"""
main.py

This module contains the main script that trains and tests the agent.
"""

from environments.gym_environment import GymEnvironment
from agents.dqn_per_agent import DQNPERAgent
from agents.dqn_agent import DQNAgent
from trainers.trainer import Trainer
import torch
import numpy as np

# Set the random seed
seed = 12844452
np.random.seed(seed)
torch.manual_seed(seed)

# Create the environment
env = GymEnvironment("CartPole-v1")

# Create the agent
agent = DQNPERAgent(4, 2)

# Train the agent
trainer = Trainer(agent, env)
trainer.train(5000)

# Test the agent
tester = Trainer(agent, env)
results = tester.test(10)
print(results)
