"""
feature_extractor.py

This file contains the FeatureExtractor class that can be used to extract a feature vector from an image.
"""

import torch
from torch import nn
from torchvision import models, transforms
import numpy as np
from typing import Any


class FeatureExtractor(nn.Module):
    """
    This class is used to extract a feature vector from an image.
    The feature vector is the output of the second to last layer of a pretrained ResNet18 model.

    Attributes:
        _resize (torchvision.transforms.Resize): A torchvision transform that resizes the smallest dimension of an image to 224 while maintaining aspect ratio.
        _model (torchvision.models.resnet.ResNet): A pretrained ResNet18 model with the last fully connected layer removed.
    """

    def __init__(self) -> None:
        """
        The constructor for the FeatureExtractor class.
        """
        super().__init__()

        # Resize the smallest dimension to 224 while maintaining aspect ratio
        self._resize = transforms.Compose([
            transforms.Resize(224),
            transforms.CenterCrop(224)
        ])

        # Load the pretrained model and remove the last fully connected layer
        self._model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
        self._model = nn.Sequential(*list(self._model.children())[:-1])

        # Set model to evaluation mode and freeze parameters
        self._model.eval()
        for param in self._model.parameters():
            param.requires_grad = False

    def forward(self, x: Any) -> torch.Tensor:
        """
        Forward pass through the model.

        Args:
            x (Any): input image.
        """
        if not isinstance(x, torch.Tensor):
            x = torch.tensor(x, dtype=torch.float32)

        # Check if the tensor has a batch dimension
        batched = x.dim() == 4
        if not batched:
            x = x.unsqueeze(0)

        # Resize with fixed aspect ratio
        x = self._resize(x)

        # Forward pass through the model
        features = self._model(x)
        features = features.view(features.size(0), -1)

        # Remove batch dimension if necessary
        if not batched:
            features = features.squeeze(0)
        return features
