"""prioritized_experience_replay_buffer.py

This module contains an implementation of Prioritized Experience Replay (PER) buffer.
"""

from schedulers import Scheduler, ExponentialDecayScheduler
from utils import Transition
from loggers import Logger
import numpy as np


class PrioritizedExperienceReplayBuffer:
    """This class is an implementation of Prioritized Experience Replay Buffer.

    Attributes:
        _max_size (int): The maximum size of the buffer.
        _full (bool): Whether the buffer is full.
        _data_index (int): The current data index.
        _batch_size (int): The size of the batch to sample.
        _data (np.ndarray): The data array.
        _priorities (np.ndarray): The priorities array.
        _weights (np.ndarray): The weights array.
        _tree_size (int): The size of the segment tree.
        _priorities_a (np.ndarray): The priorities array in the segment tree.
        _min_priorities_a (np.ndarray): The minimum priorities array in the segment tree.
        _alpha (Scheduler): The alpha value for prioritized experience replay.
        _beta (Scheduler): The beta value for prioritized experience replay.
        _epsilon (float): The epsilon value for prioritized experience replay.
    """

    def __init__(
        self,
        max_size: int = 1000000,
        batch_size: int = 32,
        alpha: float = 0.6,
        beta: Scheduler = ExponentialDecayScheduler(0.4, 1.0, 100000),
        epsilon: float = 1e-5,
    ) -> None:
        """Initializes the Experience Replay Buffer.

        Args:
            max_size (int): The maximum size of the buffer.
            batch_size (int): The size of the batch to sample.
            alpha (Scheduler): The alpha value for prioritized experience replay.
            beta (Scheduler): The beta value for prioritized experience replay.
            epsilon (float): The epsilon value for prioritized experience replay.
        """
        self._max_size = max_size
        self._full = False
        self._data_index = 0

        self._batch_size = batch_size

        # Initialize linear data arrays
        self._data = np.empty(max_size, dtype=Transition)
        self._priorities = np.zeros(max_size, dtype=np.float32)
        self._weights = np.zeros(max_size, dtype=np.float32)

        # Initialize the segment tree
        self._tree_size = 2 ** (max_size - 1).bit_length()
        self._priorities_a = np.zeros(self._tree_size * 2, dtype=np.float32)
        self._min_priorities_a = np.full(self._tree_size * 2, np.inf, dtype=np.float32)

        self._alpha = alpha
        self._beta = beta
        self._epsilon = epsilon

    def store(self, transition: Transition, error: float) -> None:
        """Stores a transition in the buffer.

        Args:
            transition (Transition): The transition to store.
            error (float): The TD error of the transition.
        """
        # Store the transition in the buffer
        self._data[self._data_index] = transition

        # Set the initial priority to the TD error
        priority = np.abs(error) + self._epsilon

        # Update the priority in the segment tree
        priority_a = np.power(priority, self._alpha)
        self._update_priority_a(self._data_index, priority_a, 0, 0, self._tree_size - 1)

        # Advance the data index
        self._advance_index()

        # Log the current buffer size
        Logger.log_scalar("experience_replay_buffer/buffer_size", len(self))

    def update(self, index: int, error: float) -> None:
        """
        Updates the priority of a transition in the buffer.

        Args:
            index (int): The index of the transition to update.
            error (float): The TD error of the transition.
        """
        # Calculate the priority
        priority = np.abs(error) + self._epsilon

        # Update the priority in the segment tree
        priority_a = np.power(priority, self._alpha)
        self._update_priority_a(index, priority_a, 0, 0, self._tree_size - 1)

    def sample(self, step: int) -> tuple[Transition, np.ndarray, list[int]]:
        """Samples a batch of transitions from the buffer.

        Args:
            step (int): The agent's current step.

        Returns:
            tuple[Transition, np.ndarray, list[int]]: A tuple with a batch of transition data, the
                sampling weights and the corresponding indices in the PER buffer.
        """
        if not self.can_sample():
            raise ValueError(
                "Not enough transitions in the buffer to sample a full batch."
            )

        # Calculate the maximum weight for normalization
        min_probability = self._min_priority_a() / self._total_priorities_a()
        max_weight = (min_probability * len(self)) ** -self._beta.value(step)

        transitions = []
        weights = []
        indices = []
        while len(transitions) < self._batch_size:
            # Sample a priority from the buffer
            priority = np.random.uniform(0, self._total_priorities_a())
            index = self._sample_priority(priority, 0, 0, self._tree_size - 1)

            # Each transition should only be sampled once
            if index in indices:
                continue

            # Store the index and transition
            indices.append(index)
            transitions.append(self._data[index])

            # Calculate the importance sampling weights
            probability = (
                self._get_priority_a_at_index(index) / self._total_priorities_a()
            )
            weight = (probability * len(self)) ** -self._beta.value(step)

            # Normalize the weight
            weight = weight / max_weight

            weights.append(weight)

        # Combine the transitions
        transitions = Transition.combine(transitions)

        # Log the current beta value
        Logger.log_scalar(
            "prioritized_experience_replay_buffer/beta", self._beta.value(step)
        )
        Logger.log_scalar("prioritized_experience_replay_buffer/weight_max", max_weight)

        return transitions, weights, indices

    def can_sample(self) -> bool:
        """Checks if the buffer has enough transitions to sample a full batch.

        Returns:
            bool: Whether the buffer has enough transitions to sample a full batch.
        """
        return len(self) >= self._batch_size

    def _update_priority_a(
        self, index: int, priority: float, node: int, start: int, end: int
    ) -> tuple[float, float]:
        """Updates the priority of a transition in the buffer, and updates the segment tree.

        Note: the start and end indices are inclusive.

        Args:
            index (int): The index of the transition to update.
            priority (float): The priority of the transition.
            node (int): The node index in the segment tree.
            start (int): The start index of the segment.
            end (int): The end index of the segment.

        Returns:
            tuple[float, float]: The sum of priorities and the minimum priority in the segment.
        """
        # Check if the node is out of range
        if index < start or index > end:
            return self._priorities_a[node], self._min_priorities_a[node]

        # Check if the node is a leaf node
        if start == end:
            # Set the priority
            self._priorities_a[node] = priority
            self._min_priorities_a[node] = priority
            return priority, priority

        # Calculate the left and right child nodes
        left_node = 2 * node + 1
        right_node = 2 * node + 2

        # Calculate the middle index
        middle = (start + end) // 2

        # Get the left and right priority sums
        left_priority, left_min_priority = self._update_priority_a(
            index, priority, left_node, start, middle
        )
        right_priority, right_min_priority = self._update_priority_a(
            index, priority, right_node, middle + 1, end
        )

        # Update the priority of the current node
        self._priorities_a[node] = left_priority + right_priority
        self._min_priorities_a[node] = min(left_min_priority, right_min_priority)
        return self._priorities_a[node], self._min_priorities_a[node]

    def _sample_priority(self, priority: float, node: int, start: int, end: int) -> int:
        """Samples a transition from the buffer based on the priority.

        Note: the start and end indices are inclusive.

        Args:
            priority (float): The priority of the transition.
            node (int): The node index in the segment tree.
            start (int): The start index of the segment.
            end (int): The end index of the segment.

        Returns:
            int: The index of the transition.
        """
        # Check if the node is a leaf node
        if start == end:
            return start

        # Calculate the left and right child nodes
        left_node = 2 * node + 1
        right_node = 2 * node + 2

        # Calculate the middle index
        middle = (start + end) // 2

        # Check if the priority is in the left child node
        if priority <= self._priorities_a[left_node]:
            return self._sample_priority(priority, left_node, start, middle)

        # The priority is in the right child node. Subtract the left node priority from the priority
        remaining_priority = priority - self._priorities_a[left_node]
        return self._sample_priority(remaining_priority, right_node, middle + 1, end)

    def _total_priorities_a(self) -> float:
        """Return the total priority of the buffer."""
        return self._priorities_a[0]

    def _min_priority_a(self) -> float:
        """Return the minimum priority**a of the buffer."""
        return self._min_priorities_a[0]

    def _get_priority_a_at_index(self, data_index: int) -> float:
        """Retrieves the priority at a specific index in the segment tree.

        Args:
            data_index (int): The index of the transition in the original data array.

        Returns:
            float: The priority at the specified index in the segment tree.
        """
        # Calculate the position of the data index in the segment tree array.
        leaf_index = self._tree_size - 1 + data_index
        return self._priorities_a[leaf_index]

    def _advance_index(self) -> None:
        """Advance the data index pointer by one.

        This is a circular buffer:
        if the pointer exceeds the maximum size, reset it to 0 and set the buffer to full.
        """
        self._data_index += 1

        # Check if the buffer is full
        if self._data_index >= self._max_size:
            self._data_index = 0
            self._full = True

    def __len__(self) -> int:
        """Return the current length of the buffer."""
        if self._full:
            return self._max_size

        return self._data_index
