"""
visual_gym_environment.py

Provides a wrapper for the a visual gymnasium environment to be interacted with by an agent.
"""

import torch
import numpy as np
import gymnasium as gym
from environments.environment import Environment
from environments.gym_environment import GymEnvironment
from vision.feature_extractor import FeatureExtractor
import cv2


class VisualGymEnvironment(Environment):
    """
    Wrapper for the visual gymnasium environment.

    This class actually wraps an instance of the GymEnvironment class.

    Attributes:
        action_size (int): The number of actions that can be taken.
        observation_size (int): The dimension of the observation space.
        continuous (bool): Whether the environment has a continuous action space.
        _environment_wrapper (GymEnvironment): The gymnasium environment wrapper.
        _feature_extractor (FeatureExtractor): The feature extractor to use.
        _prev_view (np.ndarray): The previous view of the environment.
        _render (bool): Whether to render the environment in a window.
    """

    def __init__(self, environment_name: str, render: bool = False) -> None:
        """
        Initializes the given environment

        Args:
            environment_name (str): The name of the environment.
            render (bool): Whether to render the environment in a window.
        """
        # Create the gym environment
        self._environment_wrapper = GymEnvironment(
            environment_name, render_mode="rgb_array"
        )

        self._feature_extractor = FeatureExtractor()
        self._prev_view = None
        self._render = render

    def reset(self) -> np.ndarray:
        """
        Resets the environment.

        Returns:
            np.ndarray: The initial state of the environment.
        """
        _ = self._environment_wrapper.reset()
        self._prev_view = None
        return self._get_features()

    def step(self, action: np.ndarray) -> tuple[np.ndarray, ...]:
        """
        Takes a step in the environment.

        Args:
            action (int): The action to take.

        Returns:
            tuple[np.ndarray, ...]: The observation, reward, done flag, and info.
        """
        (
            _,
            reward,
            done,
            truncated,
        ) = self._environment_wrapper.step(action)
        return self._get_features(), reward, done, truncated

    def _get_features(self) -> np.ndarray:
        """
        Gets the features from the current view of the environment.

        Returns:
            np.ndarray: The feature vector representing the view.
        """
        # Render the environment to an array
        view = self._environment_wrapper._environment.render()

        # Get the features from the view
        view_transformed = self._transform(view)
        features = self._feature_extractor(view_transformed)

        # Convert the features tensor to a numpy array
        features = features.numpy()

        return features

    def _transform(self, view: np.ndarray) -> torch.Tensor:
        """
        Transform the view to a tensor suitable for the feature extractor.

        Args:
            view (np.ndarray): The view to transform.
        Returns:
            torch.Tensor: The transformed view.
        """
        # Normalize the view
        view = view / 255.0

        # Convert the view to a tensor with format (C, H, W)
        view = torch.tensor(view, dtype=torch.float32)
        view = view.permute(2, 0, 1)

        # Add the previous view
        if self._prev_view is None:
            self._prev_view = view
        avg_view = (self._prev_view + view) / 2.0
        self._prev_view = view

        # Render the view if necessary
        if self._render:
            # Convert the tensor back to a numpy array for display
            view_np = avg_view.permute(1, 2, 0).numpy()
            view_np = (view_np * 255).astype(np.uint8)
            view_np = cv2.cvtColor(view_np, cv2.COLOR_RGB2BGR)

            # Display the image using OpenCV
            cv2.imshow("Environment View", view_np)
            cv2.waitKey(1)

        return avg_view

    @property
    def action_size(self) -> int:
        return self._environment_wrapper.action_size

    @property
    def observation_size(self) -> int:
        return self._environment_wrapper.observation_size

    @property
    def continuous(self) -> bool:
        return self._environment_wrapper.continuous
