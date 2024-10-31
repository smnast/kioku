"""
agent.py

This file defines the abstract base class for agents.
"""

from abc import ABC, abstractmethod
import numpy as np
from utils.transition import Transition


class Agent(ABC):
    """
    Abstract base class for agents.
    """

    @abstractmethod
    def act(self, observation: np.ndarray) -> np.ndarray:
        """
        Choose an action based on the current observation.

        Args:
            observation (np.ndarray): The current observation.

        Returns:
            np.ndarray: The action to take.
        """
        pass

    @abstractmethod
    def process_transition(self, transition: Transition) -> None:
        """
        Process a transition by either storing it in the memory buffer or learning on-policy.

        Args:
            transition (Transition): The transition to process.
        """
        pass

    @abstractmethod
    def learn(self) -> None:
        """
        Train the agent for one step.
        """
        pass

    @abstractmethod
    def train(self) -> None:
        """
        Set the agent to training mode.
        """
        pass

    @abstractmethod
    def test(self) -> None:
        """
        Set the agent to testing mode.
        """
        pass
