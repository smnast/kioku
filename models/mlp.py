"""
mlp.py

This module contains the definition of a Multi-Layer Perceptron (MLP) model.
"""

from utils import DEVICE
import torch
import torch.nn.functional as F
from torch import nn
from typing import Any


class MLP(nn.Module):
    """A Multi-Layer Perceptron (MLP) model.

    Attributes:
        _layers (nn.ModuleList): List of linear layers that make up the MLP.
    """

    def __init__(
        self, input_size: int, output_size: int, hidden_sizes: list[int] | None = None
    ) -> None:
        """Initializes the MLP model with a variable number of layers.

        Args:
            input_size (int): The size of the input layer.
            output_size (int): The size of the output layer.
            hidden_sizes (list[int]): The sizes of the hidden layers.
        """
        super().__init__()

        # Define the layer sizes of the ntire network
        if hidden_sizes is None:  # Avoid mutable default arguments
            hidden_sizes = []
        layer_sizes = [input_size] + hidden_sizes + [output_size]

        self.layers = nn.ModuleList()

        # Create the linear layers
        for i in range(len(layer_sizes) - 1):
            self.layers.append(nn.Linear(layer_sizes[i], layer_sizes[i + 1]))

    def forward(self, x: Any) -> torch.Tensor:
        """Defines the forward pass of the MLP.

        Args:
            x (Any): The input tensor.

        Returns:
            torch.Tensor: The output tensor after passing through the MLP.
        """
        if not isinstance(x, torch.Tensor):
            x = torch.tensor(x).to(DEVICE)

        # Pass the input through the layers
        for layer in self.layers[:-1]:
            x = F.relu(layer(x))

        # Pass the output through the final layer
        x = self.layers[-1](x)

        return x
