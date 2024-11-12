"""
main.py

This module contains the main script that trains and tests the agent.
"""

from environments.gym_environment import GymEnvironment
from agents.a2c_agent import A2CAgent
from agents.dqn_agent import DQNAgent
from trainers.trainer import Trainer
import torch
import numpy as np

# Set the random seed
seed = np.random.randint(0, 10000)
np.random.seed(seed)
torch.manual_seed(seed)

# Create the environment
env = GymEnvironment("CartPole-v1")

# Create the agent
agent = A2CAgent(env.observation_size, env.action_size)

# Train the agent
trainer = Trainer(agent, env)
trainer.train(50000)

# Test the agent
tester = Trainer(agent, env)
results = tester.test(10)
print(results)
