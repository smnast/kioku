"""
visual_gym_environment.py

Provides a wrapper for the a visual gymnasium environment to be interacted with by an agent.
"""

import numpy as np
from environments import Environment, GymEnvironment
import cv2


class VisualGymEnvironment(Environment):
    """Wrapper for the visual gymnasium environment.

    This class actually wraps an instance of the GymEnvironment class.

    Attributes:
        action_size (int): The number of actions that can be taken.
        observation_size (int | tuple[int, int, int]): The dimension of the observation space.
        continuous (bool): Whether the environment has a continuous action space.
        _environment_wrapper (GymEnvironment): The environment wrapper.
        _greyscale (bool): Whether to convert the view to greyscale.
        _resolution (tuple[int, int]): The resolution of the view.
        _prev_view (np.ndarray): The previous view of the environment.
        _render (bool): Whether to render the environment.
    """

    def __init__(self, environment_name: str, render: bool = False, greyscale: bool = False, resolution: tuple[int, int] = (64, 64)) -> None:
        """Initializes the given environment

        Args:
            environment_name (str): The name of the environment.
            render (bool): Whether to render the environment in a window.
        """
        # Create the gym environment
        self._environment_wrapper = GymEnvironment(
            environment_name, render_mode="rgb_array"
        )

        self._greyscale = greyscale
        self._resolution = resolution

        self._prev_view = None
        self._render = render

    def reset(self) -> np.ndarray:
        """Resets the environment.

        Returns:
            np.ndarray: The initial state of the environment.
        """
        _ = self._environment_wrapper.reset()
        self._prev_view = None
        return self._get_frame()

    def step(
        self, action: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Takes a step in the environment.

        Args:
            action (int): The action to take.

        Returns:
            tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]: A tuple containing:
                - The next observation (np.ndarray),
                - The reward (np.ndarray),
                - Whether the episode is done (np.ndarray),
                - Whether the episode was truncated (np.ndarray).
        """
        (
            _,
            reward,
            done,
            truncated,
        ) = self._environment_wrapper.step(action)
        return self._get_frame(), reward, done, truncated

    def _get_frame(self) -> np.ndarray:
        """Get the current frame of the environment.

        Returns:
            np.ndarray: The current frame of the environment.
        """
        # Get the view from the environment
        view = self._environment_wrapper._environment.render()
        
        # Render the environment
        if self._render:
            bgr_view = cv2.cvtColor(view, cv2.COLOR_RGB2BGR)
            cv2.imshow("Environment", bgr_view)
            cv2.waitKey(1)

        # Transform the view
        view = self._transform(view)

        # Render the agent's view (for debugging)
        # _view = view.transpose(1, 2, 0)
        # _view = np.concatenate((_view, np.zeros((*_view.shape[:2], 1), dtype=_view.dtype)), axis=2)
        # _view = cv2.resize(_view, (1024, 1024), cv2.INTER_NEAREST)
        # cv2.imshow("Environment", _view)
        # cv2.waitKey(1)

        return view

    def _transform(self, view: np.ndarray) -> np.ndarray:
        """Apply transformations to the view.

        Args:
            view (np.ndarray): The view to transform.

        Returns:
            np.ndarray: The transformed view.
        """
        # Normalize the view
        view = (view / 255.0).astype(np.float32)

        # Convert the view to greyscale
        if self._greyscale:
            view = cv2.cvtColor(view, cv2.COLOR_RGB2GRAY)

        # Resize the view
        view = cv2.resize(view, self._resolution)

        # Permute the dimensions
        if view.ndim == 3:
            view = np.transpose(view, (2, 0, 1))
        elif view.ndim == 2:
            view = np.expand_dims(view, axis=0)

        # Stack the previous view with the current view
        if self._prev_view is None:
            self._prev_view = view
        pre_stack_view = view
        view = np.concatenate((self._prev_view, view), axis=0)
        self._prev_view = pre_stack_view

        return view

    @property
    def action_size(self) -> int:
        return self._environment_wrapper.action_size

    @property
    def observation_size(self) -> int | tuple[int, int, int]:
        # Because we are stacking frames, we multiply each input channels by 2
        if self._greyscale:
            return (1 * 2, *self._resolution)
        return (3 * 2, *self._resolution)

    @property
    def continuous(self) -> bool:
        return self._environment_wrapper.continuous
