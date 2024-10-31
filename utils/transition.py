"""
transition.py

This module contains the definition of a transition object, which is used to store and retrieve transitions from experience.
"""

import numpy as np
from enum import Enum
from typing import Literal, Union

TransitionKey = Literal["observation", "action", "reward", "next_observation", "done"]


class Transition:
    """
    A transition object that stores information like observations and actions gathered from experience.

    Attributes:
        _data (dict[TransitionKey, np.ndarray]): A dictionary that stores the data of the transition.
    """

    def __init__(self, **kwargs: dict[TransitionKey, np.ndarray]) -> None:
        """
        Initializes the transition object with the given data.

        Args:
            **kwargs: The data to store in the transition object.

        Raises:
            TypeError: If any of the data is not an np.ndarray.
        """
        # Initialize the data dictionary
        self._data: dict[TransitionKey, np.ndarray] = {}

        # Store all the keys
        for key, value in kwargs.items():
            # Check if the key is a valid TransitionKey
            if key not in TransitionKey.__args__:
                raise TypeError(f"Invalid key '{key}' for transition data.")

            # Check if the value is a valid np.ndarray
            if not isinstance(value, np.ndarray):
                raise TypeError(
                    f"Expected np.ndarray for '{value}', got '{type(value)}' instead."
                )

            # Store the key-value pair in the data dictionary
            self._data[key] = value

    @classmethod
    def combine(
        self, transitions: Union[list["Transition"], np.ndarray["Transition"]]
    ) -> "Transition":
        """
        Combines a batch of transitions into a single transition object.

        Args:
            transitions (list[Transition], np.ndarray[Transition]): The batch of transitions to combine.

        Returns:
            Transition: A new transition object with the combined data.
        """
        # Collect data from all transitions in the batch
        batch_data = {}
        for key in transitions[0].keys():
            # Stack all the data along a new batch dimension (axis 0)
            batch_data[key] = np.stack([t[key] for t in transitions], axis=0)

        # Return a new Transition object containing the batch data
        return Transition(**batch_data)

    def filter(self, *keys: TransitionKey) -> "Transition":
        """
        Filters the transition object to include the specified keys.

        Args:
            *keys (TransitionKey): The keys to include in the filtered transition object.

        Returns:
            Transition: A new transition object with only the specified keys.

        Raises:
            TypeError: If any of the keys are not a valid TransitionKey.
            KeyError: If any of the keys are not in the transition object.
        """
        # Check if all keys are valid TransitionKeys
        for key in keys:
            if key not in TransitionKey.__args__:
                raise TypeError(f"Invalid key '{key}' for transition data.")

        # Check if all keys are in the data dictionary
        for key in keys:
            if key not in self._data:
                raise KeyError(f"Key '{key}' not found in transition data.")

        # Filter the data dictionary to include only the specified keys
        filtered_data = {key: self._data[key] for key in keys}
        return Transition(**filtered_data)

    def keys(self) -> list[TransitionKey]:
        """
        Returns a list of all the keys in the transition object.

        Returns:
            list[TransitionKey]: A list of all the keys in the transition object.
        """
        return list(self._data.keys())

    def __add__(self, other: "Transition") -> "Transition":
        """
        Adds the data of two transition objects.

        Args:
            other (Transition): The other transition object to add.

        Returns:
            Transition: A new transition object with the combined data.

        Raises:
            TypeError: If the other object is not a Transition.
            ValueError: If the keys of the two transitions do not match.
        """
        if not isinstance(other, Transition):
            raise TypeError(f"Expected Transition, got '{type(other)}' instead.")

        if self._data.keys() != other._data.keys():
            raise ValueError("Transition objects must have the same keys to be added.")

        combined_data = {
            key: np.concatenate((self._data[key], other._data[key]), axis=0)
            for key in self._data.keys()
        }
        return Transition(**combined_data)

    def __getitem__(
        self, keys: Union[TransitionKey, tuple[TransitionKey, ...]]
    ) -> Union[np.ndarray, tuple[np.ndarray, ...]]:
        """
        Retrieves the data stored in the transition object.

        Args:
            keys (Union[TransitionKey, tuple[TransitionKey, ...]]): The key(s) to retrieve the data.

        Returns:
            Union[np.ndarray, tuple[np.ndarray, ...]]: The data corresponding to the key(s).

        Raises:
            KeyError: If the key(s) are not in the transition object.
        """
        if isinstance(keys, tuple):
            return tuple(self[key] for key in keys)

        # The key is a single key
        key = keys

        # Check if the key is a valid TransitionKey
        if key not in TransitionKey.__args__:
            raise KeyError(f"Invalid key '{key}' for transition data.")

        # Check if the key is in the data dictionary
        if key not in self._data:
            raise KeyError(f"Key '{key}' not found in transition data.")

        # Return the data corresponding to the key
        return self._data[key]

    def __repr__(self) -> str:
        """
        Returns a string representation of the transition object.

        Returns:
            str: The string representation of the transition object.
        """
        return f"Transition({self._data})"
