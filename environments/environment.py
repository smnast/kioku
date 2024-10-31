from abc import ABC, abstractmethod
import numpy as np


class Environment(ABC):
    """
    Abstract base class for an environment.
    """

    @abstractmethod
    def reset(self) -> np.ndarray:
        """
        Resets the environment.

        Returns:
            np.ndarray: The initial state of the environment.
        """
        pass

    @abstractmethod
    def step(self, action: int) -> tuple[np.ndarray, ...]:
        """
        Takes a step in the environment.

        Args:
            action (int): The action to take.

        Returns:
            tuple[np.ndarray, float, bool, bool]: The observation, reward, done flag, and truncated flag.
        """
        pass
