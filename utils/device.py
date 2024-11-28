"""
This module sets the device to be used for PyTorch operations.

Attributes:
    DEVICE (torch.device): The device to be used for tensor operations. 
    It will be set to 'cuda' if a GPU is available, otherwise it will default to 'cpu'.
"""

import torch

# 'cuda' if a GPU is available, otherwise 'cpu'
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
