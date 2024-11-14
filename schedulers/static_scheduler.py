"""
static_scheduler.py

This file contains the implementation of the static scheduler, which always returns the same value.
"""

from schedulers import Scheduler


class StaticScheduler(Scheduler):
    """A scheduler that always returns the same value.

    Attributes:
        _value (float): The value to return.
        _test_mode (bool): Whether the scheduler is in test mode.
        _test_value (float): The value to return in test mode.
    """

    def __init__(self, value: float, test_value: float = None) -> None:
        """Initialize the static scheduler.

        Args:
            value (float): The value to return.
            test_value (float): The value to return in test mode.
        """
        self._value = value

        # Begin in training mode
        self._test_mode = False

        # If an explicit test value is provided, use it. Otherwise, default to the value.
        if test_value is not None:
            self._test_value = test_value
        else:
            self._test_value = self._value

    def value(self, step: int) -> float:
        """Get the value of the scheduler at a given step.

        Args:
            step (int): The current step.

        Returns:
            float: The value of the scheduler at the given step.
        """
        if self._test_mode:
            return self._test_value
        return self._value

    def train(self) -> None:
        """Set the scheduler to training mode."""
        self._test_mode = False

    def test(self) -> None:
        """Set the scheduler to testing mode."""
        self._test_mode = True
