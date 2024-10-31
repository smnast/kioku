"""
gym_environment.py

Provides a wrapper for the a gymnasium environment to be interacted with by an agent.
"""

from environments.environment import Environment
import gymnasium as gym
import numpy as np


class GymEnvironment(Environment):
    """
    Wrapper for the gymnasium environment.

    Attributes:
        _environment (gym.Env): The gymnasium environment.
        _seed (int): The seed for the environment.
    """

    def __init__(self, environment_name: str, render_mode: str = None) -> None:
        """
        Initializes the given environment

        Args:
            environment_name (str): The name of the environment.
            render_mode (str): The mode for rendering the environment.
        """
        self._environment = gym.make(environment_name, render_mode=render_mode)

    def reset(self) -> np.ndarray:
        """
        Resets the environment.

        Returns:
            np.ndarray: The initial state of the environment.
        """
        # Seed the environment
        random_seed = np.random.randint(0, 2**32)
        observation, _ = self._environment.reset(seed=random_seed)
        return observation

    def step(self, action: np.ndarray) -> tuple[np.ndarray, ...]:
        """
        Takes a step in the environment.

        Args:
            action (int): The action to take.

        Returns:
            tuple[np.ndarray, ...]: The observation, reward, done flag, and info.
        """
        # Retrieve the action from the numpy array
        action = action.item()

        # Take a step in the environment
        observation, reward, done, truncated, _ = self._environment.step(action)

        # Convert all the values to numpy arrays
        reward = np.array([reward], dtype=float)
        done = np.array([done], dtype=bool)
        truncated = np.array(truncated, dtype=bool)

        return observation, reward, done, truncated
