"""
n_step_buffer.py

This module contains the NStepBuffer class, which can be used to store and sample transitions
for n-step returns and General Advantage Estimation (GAE).
It discards all transitions after sampling, and only allows sampling when the buffer is full.
"""

from utils import Transition
from loggers import Logger
import numpy as np


class NStepBuffer:
    """A fixed-size buffer for n-step returns.

    It stores transitions in a buffer until it is full, and then samples a batch of transitions
    for n-step returns. It discards all transitions after sampling, and only allows sampling when
    the buffer is full (i.e. when the number of transitions stored in the buffer is equal to the
    number of steps for n-step returns).

    Attributes:
        _n_steps (int): The number of steps for n-step returns,
        _buffer (np.ndarray[Transition]): The buffer to store transitions.
        _data_index (int): The index of the data in
    """

    def __init__(self, n_steps: int = 32):
        """Initialize the NStepBuffer.

        Args:
            n_steps (int): The size of the batch to sample.
        """

        self._n_steps = n_steps

        # Initialize the buffer to store transitions
        self._data = np.empty(n_steps, dtype=Transition)
        self._data_index = 0

    def store(self, transition: Transition) -> None:
        """Store a transition in the buffer.

        Args:
            transition (Transition): The transition to store.

        Raises:
            ValueError: If the buffer is full.
        """
        # Throw an error if the buffer is full
        if self._data_index >= self._n_steps:
            raise ValueError("Buffer is full. Cannot store more transitions.")

        # Store the transition in the buffer
        self._data[self._data_index] = transition
        self._data_index += 1

        # Log the buffer size
        Logger.log_scalar("n_step_buffer/buffer_size", len(self))

    def sample(self) -> Transition:
        """Sample a batch of transitions from the buffer.

        Raises:
            ValueError: If the buffer is not full.

        Returns:
            Transition: A batch of transitions with combined data.
        """
        # Throw an error if the buffer is not full
        if not self.can_sample():
            raise ValueError("Buffer is not full. Cannot sample transitions.")

        # Combine the transitions into a single batch transition
        batch_transition = Transition.combine(self._data)

        # Reset the buffer
        self._data = np.empty(self._n_steps, dtype=Transition)
        self._data_index = 0

        return batch_transition

    def can_sample(self) -> bool:
        """Check if the buffer can sample a batch of transitions.

        Returns:
            bool: Whether the buffer can sample a batch of transitions.
        """
        return len(self) == self._n_steps

    def __len__(self) -> int:
        """Get the number of transitions stored in the buffer.

        Returns:
            int: The number of transitions stored in the buffer.
        """
        return self._data_index
