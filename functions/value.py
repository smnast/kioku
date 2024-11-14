"""
value.py

This module contains the definition of a value function approximator class, Value.
"""

import torch
from torch import nn, optim
from models.mlp import MLP
from schedulers.scheduler import Scheduler
from schedulers.static_scheduler import StaticScheduler
from loggers.logger import Logger


class Value:
    """A simple single-network value function approximator.

    This network is typically used for value functions in policy gradient methods like A2C.

    Attributes:
        _network (MLP): The online network used for value estimation.
        _optimizer (optim.Adam): The optimizer.
        _learning_rate (Scheduler): The learning rate scheduler.
        _gradient_clipping (float): The maximum gradient norm for clipping.
    """

    def __init__(
        self,
        input_size: int,
        output_size: int,
        hidden_sizes: list[int] = [],
        learning_rate: Scheduler = StaticScheduler(3e-4),
        gradient_clipping: float = 0.5,
    ) -> None:
        """Initializes the Value network.

        Args:
            input_size (int): The size of the input tensor.
            output_size (int): The size of the output tensor.
            hidden_sizes (list[int]): Sizes of hidden layers.
            learning_rate (Scheduler): The learning rate of the optimizer.
        """
        self._network = MLP(input_size, output_size, hidden_sizes)
        self._optimizer = optim.Adam(
            self._network.parameters(), lr=learning_rate.value(0)
        )
        self._learning_rate = learning_rate
        self._gradient_clipping = gradient_clipping

    def predict(self, x: torch.Tensor) -> torch.Tensor:
        """Computes the value estimate for a given input.

        Args:
            x (torch.Tensor): The input tensor.

        Returns:
            torch.Tensor: The predicted value estimate.
        """
        return self._network(x)

    def optimize(self, loss: torch.Tensor, step: int) -> None:
        """Optimizes the Value network.

        Args:
            loss (torch.Tensor): The loss tensor to backpropagate.
            step (int): The current training step.
        """
        # Update the optimizer learning rate
        self._learning_rate.adjust(self._optimizer, step)

        # Optimize the network
        self._optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(
            self._network.parameters(), max_norm=self._gradient_clipping
        )
        self._optimizer.step()

        # Log the learning rate
        Logger.log_scalar("value/learning_rate", self._learning_rate.value(step))

        # Log the average gradient magnitude
        avg_gradient = sum(
            p.grad.abs().mean() for p in self._network.parameters()
        ) / len(list(self._network.parameters()))
        Logger.log_scalar("value/gradient", avg_gradient)

    def train(self) -> None:
        """Sets the Value network to training mode."""
        self._network.train()
        self._learning_rate.train()

    def test(self) -> None:
        """Sets the Value network to evaluation mode."""
        self._network.eval()
        self._learning_rate.test()
