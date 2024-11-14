"""
agent.py

This file defines the abstract base class for agents.
"""

import numpy as np
from utils.transition import Transition
from abc import ABC, abstractmethod


class Agent(ABC):
    """Abstract base class for agents. Defines the common interface for all agent classes.

    An agent interacts with an environment by observing states, taking actions, and learning from
    transitions between states. Subclasses must implement the abstract methods to define specific
    agent behavior.
    """

    @abstractmethod
    def act(
        self, observation: np.ndarray, state: dict | None = None
    ) -> tuple[np.ndarray, dict | None]:
        """Choose an action based on the current observation.

        Args:
            observation (np.ndarray): The current observation from the environment.
            state (dict | None, optional): The inner state of the agent. Defaults to None.

        Returns:
            tuple[np.ndarray, dict | None]: A tuple containing:
                - The action to take (np.ndarray).
                - The new state of the agent (dict | None).
        """
        pass

    @abstractmethod
    def process_transition(self, transition: Transition) -> None:
        """Process a transition by either storing it in memory or learning on-policy.

        Args:
            transition (Transition): The transition object containing the state, action, reward,
                and next state.
        """
        pass

    @abstractmethod
    def learn(self) -> None:
        """Train the agent for one step based on its current knowledge and experience."""
        pass

    @abstractmethod
    def train(self) -> None:
        """Set the agent to training mode, where it actively learns from interactions with the
        environment.
        """
        pass

    @abstractmethod
    def test(self) -> None:
        """Set the agent to testing mode, where it behaves according to the learned policy without
        further training.
        """
        pass
