"""
a2c_agent.py

This module contains the A2CAgent class,
which is an implementation of the Advantage Actor-Critic (A2C) algorithm.
"""

from agents import Agent
from memory import NStepBuffer
from functions import DiscreteActor, Value
from schedulers import Scheduler, StaticScheduler
from utils import Transition, DEVICE
from loggers import Logger
import torch
import torch.nn.functional as F
import numpy as np


class A2CAgent(Agent):
    """An implementation of the Advantage Actor-Critic (A2C) algorithm.

    Attributes:
        _actor (DiscreteActor): The actor network.
        _critic (Value): The critic network.
        _n_steps (int): The number of steps to use for n-step returns.
        _n_step_buffer (NStepBuffer): The buffer to store n-step transitions.
        _relevant_keys (list[str]): The keys to keep when processing transitions.
        _gamma (float): The discount factor for future rewards.
        _lambda (float): The GAE lambda parameter.
        _critic_coefficient (float): The coefficient for the critic loss.
        _normalize_advantages (bool): Whether to normalize the advantages.
        _step (int): The current training step.
    """

    def __init__(
        self,
        observation_size: int,
        num_actions: int,
        actor_hidden_sizes: list[int] = [128, 128],
        critic_hidden_sizes: list[int] = [128, 128],
        actor_learning_rate: Scheduler = StaticScheduler(7e-4, 0),
        critic_learning_rate: Scheduler = StaticScheduler(7e-4, 0),
        critic_coefficient: float = 0.5,
        normalize_advantages: bool = False,
        gamma: float = 0.995,
        lambda_: float = 0.95,
        n_steps: int = 20,
        gradient_clipping: float = 0.5,
    ) -> None:
        """Initialize the A2C agent.

        Args:
            observation_size (int): The size of the observation space.
            num_actions (int): The number of actions in the action space.
            actor_hidden_sizes (list[int]): The sizes of the hidden layers for the actor network.
            critic_hidden_sizes (list[int]): The sizes of the hidden layers for the critic network.
            actor_learning_rate (Scheduler): The learning rate of the actor's optimizer.
            critic_learning_rate (Scheduler): The learning rate of the critic's optimizer.
            critic_coefficient (float): The coefficient for the critic loss.
            normalize_advantages (bool): Whether to normalize the advantages.
            gamma (float): The discount factor for future rewards.
            lambda_ (float): The GAE lambda parameter.
            n_steps (int): The number of steps to use for n-step returns.
            gradient_clipping (float): The max gradient norm for the agent's models.
        """
        self._actor = DiscreteActor(
            observation_size,
            num_actions,
            actor_hidden_sizes,
            learning_rate=actor_learning_rate,
            gradient_clipping=gradient_clipping,
        )
        self._critic = Value(
            observation_size,
            1,
            critic_hidden_sizes,
            learning_rate=critic_learning_rate,
            gradient_clipping=gradient_clipping,
        )

        self._n_steps = n_steps
        self._n_step_buffer = NStepBuffer(self._n_steps)
        self._relevant_keys = [
            "observation",
            "reward",
            "next_observation",
            "done",
            "action_log_prob",
        ]

        self._gamma = gamma
        self._lambda = lambda_
        self._critic_coefficient = critic_coefficient
        self._normalize_advantages = normalize_advantages

        self._step = 0

    def act(
        self, observation: np.ndarray, state: dict | None = None
    ) -> tuple[np.ndarray, dict | None]:
        """Choose an action based on the current observation.

        Args:
            observation (np.ndarray): The current observation.
            state (dict | None): The state of the agent.

        Returns:
            tuple[np.ndarray, dict | None]: The action to take, and the new state of the agent.
        """
        # Update the current step
        self._step += 1

        # Get the action from the actor
        action, action_log_prob = self._actor.act(observation)
        action_log_prob = action_log_prob.unsqueeze(-1)

        # Convert the action to a numpy array
        action = action.cpu().numpy()

        return action, {
            "action_log_prob": action_log_prob,
        }

    def process_transition(self, transition: Transition) -> None:
        """Process a transition by storing it in the buffer.

        Args:
            transition (Transition): The transition to process.
        """
        transition.filter(self._relevant_keys)
        self._n_step_buffer.store(transition)

    def learn(self) -> None:
        """Train the agent for one step."""
        if not self._n_step_buffer.can_sample():
            return

        # Get the batch of transitions
        batch_transition = self._n_step_buffer.sample()
        observation, reward, next_observation, done, action_log_prob = batch_transition[
            self._relevant_keys
        ]

        # Convert to tensors when necessary
        observation = torch.tensor(observation, dtype=torch.float32).to(DEVICE)
        reward = torch.tensor(reward, dtype=torch.float32).squeeze().to(DEVICE)
        next_observation = torch.tensor(next_observation, dtype=torch.float32).to(DEVICE)
        done = torch.tensor(done, dtype=torch.bool).squeeze().to(DEVICE)

        # Squeeze the action log probabilities
        action_log_prob = action_log_prob.squeeze()

        # Compute the value estimates
        value_estimate = self._critic.predict(observation).squeeze()
        next_value_estimate = self._critic.predict(next_observation).squeeze()

        # Compute GAE advantages
        advantage = self._compute_gae(
            reward, value_estimate.detach(), next_value_estimate.detach(), done
        )

        # (Optionally) normalize the advantages
        if self._normalize_advantages:
            advantage = (advantage - advantage.mean()) / (advantage.std() + 1e-8)

        # Compute targets for the critic (returns)
        returns = advantage + value_estimate.detach()

        # Compute the critic loss
        critic_loss = self._critic_coefficient * F.mse_loss(value_estimate, returns)

        # Compute the actor loss
        actor_loss = -torch.mean(action_log_prob * advantage)

        # Update the actor and critic
        self._actor.optimize(actor_loss, self._step)
        self._critic.optimize(critic_loss, self._step)

        # Log the training process
        Logger.log_scalar("a2c_agent/actor_loss", actor_loss)
        Logger.log_scalar("a2c_agent/critic_loss", critic_loss)
        Logger.log_scalar("a2c_agent/advantage/max", advantage.max())
        Logger.log_scalar("a2c_agent/advantage/min", advantage.min())
        Logger.log_scalar("a2c_agent/advantage/mean", advantage.mean())
        Logger.log_scalar("a2c_agent/value_estimate/max", value_estimate.max())
        Logger.log_scalar("a2c_agent/value_estimate/min", value_estimate.min())
        Logger.log_scalar("a2c_agent/value_estimate/mean", value_estimate.mean())
        Logger.log_scalar(
            "a2c_agent/next_value_estimate/max", next_value_estimate.max()
        )
        Logger.log_scalar(
            "a2c_agent/next_value_estimate/min", next_value_estimate.min()
        )
        Logger.log_scalar(
            "a2c_agent/next_value_estimate/mean", next_value_estimate.mean()
        )
        Logger.log_scalar("a2c_agent/returns/max", returns.max())
        Logger.log_scalar("a2c_agent/returns/min", returns.min())
        Logger.log_scalar("a2c_agent/returns/mean", returns.mean())

    def _compute_gae(
        self,
        rewards: torch.Tensor,
        values: torch.Tensor,
        next_values: torch.Tensor,
        dones: torch.Tensor,
    ) -> torch.Tensor:
        """Computes Generalized Advantage Estimation (GAE).

        Args:
            rewards (torch.Tensor): The rewards for each step.
            values (torch.Tensor): Value estimates from the critic for each step.
            next_values (torch.Tensor): Next step value estimates from the critic.
            dones (torch.Tensor): Whether each step was terminal.

        Returns:
            torch.Tensor: The computed advantages.
        """
        # Compute TD errors
        td_errors = rewards + self._gamma * next_values * (~dones) - values

        # GAE computation
        advantage = 0
        advantages = torch.zeros_like(rewards, dtype=torch.float32)
        for t in reversed(range(self._n_steps)):
            advantage = self._gamma * self._lambda * advantage + td_errors[t]
            advantages[t] = advantage

        return advantages

    def train(self) -> None:
        """Set the agent to training mode."""
        self._actor.train()
        self._critic.train()

    def test(self) -> None:
        """Set the agent to testing mode."""
        self._actor.test()
        self._critic.test()
