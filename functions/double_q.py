"""
double_q.py

This module contains the definition of a simple double q network.
One network is the online network and the other is the target network.
"""

import torch
from torch import nn, optim
from models.mlp import MLP
from schedulers.scheduler import Scheduler
from schedulers.static_scheduler import StaticScheduler
from loggers.logger import Logger


class DoubleQ:
    """
    A simple Double Q network.

    Attributes:
        _online_network (MLP): The online network.
        _target_network (MLP): The target network.
        _optimizer (optim.Adam): The optimizer.
        _transition_rate (float): The rate at which the target network transitions to the online network.
    """

    def __init__(
        self,
        input_size: int,
        output_size: int,
        hidden_sizes: list[int] = [],
        learning_rate: Scheduler = StaticScheduler(3e-4),
        transition_rate: float = 0.005,
    ) -> None:
        """
        Initializes the Double Q network.

        Args:
            input_size (int): The size of the input tensor.
            output_size (int): The size of the output tensor.
            learning_rate (Scheduler): The learning rate of the optimizer.
            transition_rate (float): The rate at which the target network transitions to the online network.
        """
        self._online_network = MLP(input_size, output_size, hidden_sizes)
        self._target_network = MLP(input_size, output_size, hidden_sizes)
        self._copy_online_to_target()

        self._optimizer = optim.Adam(
            self._online_network.parameters(), lr=learning_rate.value(0)
        )

        self._learning_rate = learning_rate
        self._transition_rate = transition_rate

    def predict(self, x: torch.Tensor, target: bool = False) -> torch.Tensor:
        """
        Defines the q prediction of the Double Q network.

        Args:
            x (torch.Tensor): The input tensor.
            target (bool): Whether to use the target network or the online network.
        Returns:
            torch.Tensor: The output tensor after passing through the Double Q network.
        """
        if target:
            with torch.no_grad():
                return self._target_network(x)
        return self._online_network(x)

    def optimize(self, loss: torch.Tensor, step: int) -> None:
        """
        Optimize the Double Q network.

        Args:
            loss (torch.Tensor): The loss tensor to backpropagate.
            step (int): The current step.
        """
        # Update the optimizer learning rate
        for param_group in self._optimizer.param_groups:
            param_group["lr"] = self._learning_rate.value(step)

        # Optimize the online network
        self._optimizer.zero_grad()
        loss.backward()
        self._optimizer.step()

        # Update the target network
        self._update_target_network()

        # Log the learning rate
        Logger.log_scalar("double_q/learning_rate", self._learning_rate.value(step))

        # Log the gradients
        avg_gradient = 0
        for param in self._online_network.parameters():
            avg_gradient += param.grad.abs().mean()
        avg_gradient /= len(list(self._online_network.parameters()))
        Logger.log_scalar("double_q/gradient", avg_gradient)

    def train(self) -> None:
        """
        Sets the Double Q network to training mode.
        """
        self._online_network.train()
        self._target_network.train()
        self._learning_rate.train()

    def test(self) -> None:
        """
        Sets the Double Q network to evaluation mode.
        """
        self._online_network.eval()
        self._target_network.eval()
        self._learning_rate.test()

    def _update_target_network(self) -> None:
        """
        Updates the target network with the online network's parameters.
        """
        target_state_dict = self._target_network.state_dict()
        online_state_dict = self._online_network.state_dict()
        for key in target_state_dict:
            target_state_dict[key] = self._transition_rate * online_state_dict[key] + (1 - self._transition_rate) * target_state_dict[key]
        self._target_network.load_state_dict(target_state_dict)

    def _copy_online_to_target(self) -> None:
        """
        Copies the online network's parameters to the target network.
        """
        self._target_network.load_state_dict(self._online_network.state_dict())
