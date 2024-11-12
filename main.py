"""
main.py

This module contains the main script that trains and tests the agent.
"""

from environments.gym_environment import GymEnvironment
from agents.a2c_agent import A2CAgent
from trainers.trainer import Trainer
import torch
import numpy as np

# Set the random seed
seed = 42
np.random.seed(seed)
torch.manual_seed(seed)

# Create the environment
env = GymEnvironment("LunarLander-v2")
render_env = GymEnvironment("LunarLander-v2", render_mode="human")

# Create the agent
agent = A2CAgent(env.observation_size, env.action_size)

# Train the agent
trainer = Trainer(agent, env)
trainer.train(200000)

# Test the agent
tester = Trainer(agent, render_env)
results = tester.test(10)
print(results)
