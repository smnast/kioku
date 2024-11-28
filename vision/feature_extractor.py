"""
feature_extractor.py

This file contains the FeatureExtractor class that can be used to extract a feature vector from an image.
"""

from utils import DEVICE
import torch
from torch import nn
import torch.nn.functional as F
from typing import Any


class FeatureExtractor(nn.Module):
    """
    This class is used to extract a feature vector from an image.

    Attributes:
        output_size (int): The size of the output feature vector.
        _input_height (int): The height of the input image.
        _input_width (int): The width of the input image.
        _input_channels (int): The number of channels in the input image.
        _conv_layers (nn.ModuleList): List of convolutional layers.
        _pool_layers (nn.ModuleList): List of pooling layers.
        _conv_output_height (int): The height of the output from the convolutional layers.
        _conv_output_width (int): The width of the output from the convolutional layers.
        _fc (nn.Linear): Fully connected layer to produce the feature vector.
    """

    def __init__(
        self,
        input_shape: tuple[int, int, int],
        output_size: int,
        conv_channels: list[int],
        kernel_sizes: list[int],
        strides: list[int],
        paddings: list[int],
        pool_kernel_sizes: list[int],
        pool_strides: list[int],
    ) -> None:
        """
        The constructor for the FeatureExtractor class.

        Args:
            input_shape (tuple[int, int, int]): Shape of the input image (height, width, channels).
            output_size (int): The desired size of the output feature vector.
            conv_channels (list[int]): List of number of channels for each convolutional layer.
            kernel_sizes (list[int]): List of kernel sizes for each convolutional layer.
            strides (list[int]): List of strides for each convolutional layer.
            paddings (list[int]): List of padding values for each convolutional layer.
            pool_kernel_sizes (list[int]): List of kernel sizes for pooling layers.
            pool_strides (list[int]): List of strides for pooling layers.
        """
        super(FeatureExtractor, self).__init__()

        # Save the output size
        self.output_size = output_size

        # The input shape has the format (height, width, channels).
        self._input_channels, self._input_height, self._input_width = input_shape

        # Define the convolutional and pooling layers
        self._conv_layers = nn.ModuleList()
        self._pool_layers = nn.ModuleList()
        in_channels = self._input_channels
        for (
            out_channels,
            kernel_size,
            stride,
            pad,
            pool_kernel_size,
            pool_stride,
        ) in zip(
            conv_channels,
            kernel_sizes,
            strides,
            paddings,
            pool_kernel_sizes,
            pool_strides,
        ):
            # Convolutional layer
            self._conv_layers.append(
                nn.Conv2d(in_channels, out_channels, kernel_size, stride, pad)
            )
            # Pooling layer
            self._pool_layers.append(nn.MaxPool2d(pool_kernel_size, pool_stride))
            in_channels = out_channels

        # Calculate the output size after the convolution and pooling layers
        self._conv_output_height = self._input_height
        self._conv_output_width = self._input_width
        for stride, kernel_size, pad, pool_stride, pool_kernel_size in zip(
            strides, kernel_sizes, paddings, pool_strides, pool_kernel_sizes
        ):
            # Apply convolution output size calculation
            self._conv_output_height = (
                self._conv_output_height + 2 * pad - kernel_size
            ) // stride + 1
            self._conv_output_width = (
                self._conv_output_width + 2 * pad - kernel_size
            ) // stride + 1
            # Apply pooling output size calculation
            self._conv_output_height = (
                self._conv_output_height - pool_kernel_size
            ) // pool_stride + 1
            self._conv_output_width = (
                self._conv_output_width - pool_kernel_size
            ) // pool_stride + 1

        # Define the fully connected layer to produce the feature vector
        self._fc = nn.Linear(
            in_channels * self._conv_output_height * self._conv_output_width,
            output_size,
        )

    def forward(self, x: Any) -> torch.Tensor:
        """
        Forward pass through the model.

        Args:
            x (Any): Input image tensor of shape (batch_size, channels, height, width).

        Returns:
            torch.Tensor: The output feature vector of shape (batch_size, output_size).
        """
        # Convert input to a tensor
        if not isinstance(x, torch.Tensor):
            x = torch.tensor(x, dtype=torch.float32).to(DEVICE)

        # Add a batch dimension if necessary
        added_batch_dim = False
        if len(x.shape) == 3:
            added_batch_dim = True
            x = x.unsqueeze(0)

        # Apply the convolutional and pooling layers
        for conv_layer, pool_layer in zip(self._conv_layers, self._pool_layers):
            x = F.relu(conv_layer(x))
            x = pool_layer(x)

        # Flatten the output from the conv layers to feed into the fully connected layer
        x = x.reshape(x.size(0), -1)

        # Apply the fully connected layer to get the feature vector
        x = self._fc(x)

        # Remove the batch dimension if it was added
        if added_batch_dim:
            x = x.squeeze(0)

        return x
