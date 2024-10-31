"""
scheduler.py

This file contains the abstract class for a scheduler.
"""

from abc import ABC, abstractmethod


class Scheduler:
    """
    An abstract class for a scheduler.
    """

    @abstractmethod
    def value(self, step: int) -> float:
        """
        Get the value of the scheduler at a given step.

        Args:
            step (int): The current step.

        Returns:
            float: The value of the scheduler at the given step.
        """
        pass

    @abstractmethod
    def train(self) -> None:
        """
        Set the scheduler to training mode.
        """
        pass

    @abstractmethod
    def test(self) -> None:
        """
        Set the scheduler to testing mode.
        """
        pass
