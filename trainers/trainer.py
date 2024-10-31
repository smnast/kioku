"""
trainer.py

This file contains the Trainer class, which is responsible for training an agent on a given environment.
"""

from agents.agent import Agent
from environments.environment import Environment
from utils.transition import Transition
from typing import Any
from loggers.logger import Logger

class Trainer:
    """
    The Trainer class trains an agent on a given environment.

    Attributes:
        _agent (Agent): The agent to train.
        _environment (Environment): The environment to train the agent on.
    """

    def __init__(self, agent: Agent, environment: Environment) -> None:
        """
        Initializes the Trainer class.

        Args:
            agent (Agent): The agent to train.
            environment (Environment): The environment to train the agent on.
        """
        self._agent = agent
        self._environment = environment

    def train(self, num_steps: int) -> None:
        """
        Trains the agent for a specified number of steps.

        Args:
            num_steps (int): The number of steps to train the agent for.
        """
        # Put the agent in training mode
        self._agent.train()

        episode_complete = False
        episode_reward = 0
        for step in range(1, num_steps+1):
            if episode_complete or step == 1:
                # Reset the environment
                observation = self._environment.reset()

            if episode_complete:
                # Log the total reward for the episode
                Logger.log_scalar("train/episode_reward", episode_reward)

                # Reset the episode stats
                episode_complete = False
                episode_reward = 0

            # Take a step in the environment
            action = self._agent.act(observation)
            next_observation, reward, done, truncated = self._environment.step(action)

            # Process transition
            transition = Transition(
                observation=observation,
                action=action,
                reward=reward,
                next_observation=next_observation,
                done=done,
            )
            self._agent.process_transition(transition)

            # Update the next state
            observation = next_observation

            # Train the agent
            self._agent.learn()

            # End the episode if done or truncated is True
            if done.any() or truncated.any():
                episode_complete = True

            # Log the training progress
            Logger.log_scalar("train/reward", reward)
            Logger.log_scalar("train/action", action)

            # Accumulate total reward for the episode
            episode_reward += reward

    def test(self, num_episodes: int) -> dict[str, Any]:
        """
        Tests the agent for a specified number of episodes and returns the results.

        Returns:
            dict[str, Any]: The results dictionary containing information about the testing.
        """
        # Put the agent in testing mode
        self._agent.test()

        # Store the reward for each episode
        episode_rewards = []

        for _ in range(num_episodes):
            # Reset the environment
            observation = self._environment.reset()

            # Initialize the episode stats
            episode_complete = False
            episode_reward = 0

            # Run the episode
            while not episode_complete:
                # Take a step in the environment
                action = self._agent.act(observation)
                next_observation, reward, done, truncated = self._environment.step(
                    action
                )

                # Update the next state
                observation = next_observation

                # Add the reward
                episode_reward += reward

                # End the episode if done or truncated is True
                if done.any() or truncated.any():
                    episode_complete = True

                # Log the testing progress
                Logger.log_scalar("test/reward", reward)
                Logger.log_scalar("test/action", action)

            episode_rewards.append(episode_reward)

            # Log the total reward for the episode
            Logger.log_scalar("test/episode_reward", episode_reward)

        # Compute the average reward per episode
        avg_reward = sum(episode_rewards) / num_episodes

        return {"avg_reward": avg_reward, "episode_rewards": episode_rewards}
