"""
discrete_actor.py

This module contains the DiscreteActor class, which is an implementation of an actor for discrete
action spaces.
"""

import torch
import torch.nn.functional as F
from torch.distributions import Categorical
import numpy as np
from models.mlp import MLP
from schedulers.scheduler import Scheduler
from schedulers.static_scheduler import StaticScheduler
from loggers.logger import Logger


class DiscreteActor:
    """An implementation of an actor for discrete action spaces.

    Attributes:
        _model (MLP): The model used to predict the action probabilities.
        _learning_rate (Scheduler): The learning rate scheduler.
        _optimizer (torch.optim.Adam): The optimizer.
        _gradient_clipping (float): The maximum gradient norm for clipping
    """

    def __init__(
        self,
        observation_size: int,
        num_actions: int,
        hidden_sizes: list[int] = [32, 32],
        learning_rate: Scheduler = StaticScheduler(3e-4, 0),
        gradient_clipping: float = 0.5,
    ) -> None:
        """
        Initialize the DiscreteActor.

        Args:
            observation_size (int): The size of the observation space.
            num_actions (int): The number of actions in the action space.
            hidden_sizes (list[int]): The sizes of the hidden layers.
        """

        # Create the model to predict the action probabilities
        self._model = MLP(observation_size, num_actions, hidden_sizes)
        self._learning_rate = learning_rate
        self._optimizer = torch.optim.Adam(
            self._model.parameters(), lr=self._learning_rate.value(0)
        )

        self._gradient_clipping = gradient_clipping

    def act(self, observation: np.ndarray) -> tuple[torch.Tensor, torch.Tensor]:
        """Select an action based on the observation.

        Args:
            observation (np.ndarray): The observation to predict the action probabilities for.

        Returns:
            tuple[torch.Tensor, torch.Tensor]: The action and the log probability of the action.
        """
        # Get the action probabilities from the model
        logits = self._model(observation)

        # Create a distribution from the action probabilities
        action_dist = Categorical(logits=logits)
        action = action_dist.sample()
        action_log_prob = action_dist.log_prob(action)

        return action, action_log_prob

    def optimize(self, loss: torch.Tensor, step: int) -> None:
        """Optimize the actor model.

        Args:
            actor_loss (torch.Tensor): The loss of the actor model.
            step (int): The current step.
        """
        # Update the optimizer learning rate
        self._learning_rate.adjust(self._optimizer, step)

        # Optimize the model
        self._optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(
            self._model.parameters(), max_norm=self._gradient_clipping
        )
        self._optimizer.step()

        # Log the learning rate
        Logger.log_scalar("actor/learning_rate", self._learning_rate.value(step))

        # Log the model gradients
        average_grad_norm = np.mean(
            [
                torch.norm(param.grad).item()
                for param in self._model.parameters()
                if param.grad is not None
            ]
        )
        Logger.log_scalar("actor/gradient_norm", average_grad_norm)

    def train(self) -> None:
        """Set the actor model to training mode."""
        self._model.train()
        self._learning_rate.train()

    def test(self) -> None:
        """Set the actor model to evaluation mode."""
        self._model.eval()
        self._learning_rate.test()
