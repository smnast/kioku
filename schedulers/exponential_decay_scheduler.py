"""
exponential_decay_scheduler.py

This file contains the implementation of the exponential decay scheduler, which decays a value
exponentially over time.
"""

import numpy as np
from schedulers.scheduler import Scheduler


class ExponentialDecayScheduler(Scheduler):
    """An exponential decay scheduler.

    Attributes:
        _begin (float): The initial value of the decay.
        _end (float): The final value of the decay.
        _time (float): The time over which the decay occurs.
        _test_mode (bool): Whether the scheduler is in test mode.
        _test_value (float): The value to return in test mode
    """

    def __init__(
        self, begin: float, end: float, time: float, test_value: float = None
    ) -> None:
        """Initialize the exponential decay scheduler.

        Args:
            begin (float): The initial value of the decay.
            end (float): The final value of the decay.
            time (float): The time over which the decay occurs.
            test_value (float): The value to return in test mode.
        """
        self._begin = begin
        self._end = end
        self._time = time

        # Begin in training mode
        self._test_mode = False
        
        # If an explicit test value is provided, use it. Otherwise, default to the end value.
        self._test_value = self._end if test_value is None else test_value

    def value(self, step: int) -> float:
        """Get the value of the decay at a given step.

        Args:
            step (int): The current step.

        Returns:
            float: The value of the decay at the given step.
        """
        # If in test mode, return the test value, otherwise return the decayed value
        if self._test_mode:
            return self._test_value
        if step >= self._time:
            return self._end
        return self._begin * (self._end / self._begin) ** (step / self._time)

    def train(self) -> None:
        """Set the scheduler to training mode."""
        self._test_mode = False

    def test(self) -> None:
        """Set the scheduler to testing mode."""
        self._test_mode = True
