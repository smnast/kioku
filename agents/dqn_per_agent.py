"""
dqn_per_agent.py

This file contains the implementation of the DQN agent with a
Prioritized Experience Replay (PER) buffer.
"""

from agents import Agent
from utils import Transition
from memory import PrioritizedExperienceReplayBuffer
from functions import DoubleValue
from schedulers import Scheduler, ExponentialDecayScheduler, StaticScheduler
from loggers import Logger
import torch
import numpy as np


class DQNPERAgent(Agent):
    """A Deep Q-Network (DQN) agent using a Prioritized Experience Replay (PER) buffer.

    Attributes:
        _epsilon (Scheduler): The probability of taking a random action (exploration).
        _gamma (float): The discount factor for future rewards.
        _model (DoubleQ): The Q model used to estimate action values.
        _num_actions (int): The number of actions the agent can take.
        _batch_size (int): The number of experiences used per training batch.
        _memory (ExperienceReplayBuffer): The buffer storing past experiences for training.
        _relevant_keys (list[str]): The keys of the relevant information in a transition.
        _steps (int): The total number of steps the agent has taken during training.
    """

    def __init__(
        self,
        observation_size: int,
        num_actions: int,
        hidden_sizes: list[int] = [32, 32],
        learning_rate: Scheduler = StaticScheduler(1e-2, 0),
        epsilon: Scheduler = ExponentialDecayScheduler(1, 0.01, 5000, 0),
        gamma: float = 0.99,
        transition_rate: float = 0.005,
        memory_size: int = 5000,
        batch_size: int = 32,
        alpha: float = 0.4,
        beta: Scheduler = ExponentialDecayScheduler(0.6, 1.0, 5000),
        buffer_epsilon: float = 1e-3,
        gradient_clipping: float = 0.5,
    ) -> None:
        """Initialize the DQN agent.

        Args:
            observation_size (int): The size of the observation.
            num_actions (int): The number of actions the agent can take.
            hidden_sizes (list[int]): The sizes of the models' hidden layers.
            learning_rate (Scheduler): The learning rate scheduler.
            epsilon (Scheduler): The epsilon value scheduler.
            gamma (float): The discount factor for future rewards.
            transition_rate (float): The rate at which the model is updated.
            memory_size (int): The maximum size of the memory buffer.
            batch_size (int): The batch size used for training.
            alpha (float): The weight of the importance-sampling correction in PER.
            beta (Scheduler): The scheduler for adjusting the importance-sampling exponent.
            buffer_epsilon (float): A small value to avoid dividing by zero in importance-sampling.
            gradient_clipping (float): The max gradient norm for the agent's models.
        """
        self._epsilon = epsilon
        self._gamma = gamma

        self._model = DoubleValue(
            input_size=observation_size,
            output_size=num_actions,
            hidden_sizes=hidden_sizes,
            learning_rate=learning_rate,
            transition_rate=transition_rate,
            gradient_clipping=gradient_clipping,
        )
        self._num_actions = num_actions

        self._batch_size = batch_size
        self._memory = PrioritizedExperienceReplayBuffer(
            memory_size, batch_size, alpha, beta, buffer_epsilon
        )
        self._relevant_keys = [
            "observation",
            "action",
            "reward",
            "next_observation",
            "done",
        ]

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

        # Choose an action
        chosen_action = None
        epsilon_value = self._epsilon.value(self._step)
        if np.random.rand() < epsilon_value:
            # Select a random action
            chosen_action = np.random.randint(self._num_actions)
        else:
            # Select the action with the highest predicted Q-value
            with torch.no_grad():
                q_values = self._model.predict(observation)
                q_values = q_values.cpu().numpy()
            chosen_action = np.argmax(q_values)

        # Store the action in a numpy array
        chosen_action = np.array([chosen_action])

        # Log the epsilon value
        Logger.log_scalar("dqn_agent/epsilon", epsilon_value)

        return chosen_action, state

    def process_transition(self, transition: Transition) -> None:
        """Process a transition by either storing it in the memory buffer or learning on-policy.

        Args:
            transition (Transition): The transition to process.
        """
        transition = transition.filter(self._relevant_keys)

        # Compute TD error of transition
        td_error, _, _ = self._compute_td_error(transition)

        # Store the transition in the memory buffer with its TD error
        self._memory.store(transition, td_error.item())

    def learn(self) -> None:
        """Train the agent on a batch of experiences."""
        if not self._memory.can_sample():
            return

        # Get the batch of transitions
        batch_transition, sampling_weights, buffer_indices = self._memory.sample(
            self._step
        )

        # Compute TD error and loss
        td_error, current_q_values, target_q_values = self._compute_td_error(
            batch_transition
        )
        sampling_weights = torch.tensor(sampling_weights, dtype=torch.float32)
        q_loss = (torch.square(td_error) * sampling_weights).mean()

        # Optimize the model
        self._model.optimize(q_loss, self._step)

        # Update the priorities in the memory buffer
        for batch_index, buffer_index in enumerate(buffer_indices):
            self._memory.update(buffer_index, td_error[batch_index].item())

        # Log the learning process
        Logger.log_scalar("dqn_agent/loss", q_loss.item())
        Logger.log_scalar("dqn_agent/pred_q_value/max", current_q_values.max().item())
        Logger.log_scalar("dqn_agent/pred_q_value/min", current_q_values.min().item())
        Logger.log_scalar("dqn_agent/pred_q_value/mean", current_q_values.mean().item())
        Logger.log_scalar("dqn_agent/target_q_value/max", target_q_values.max().item())
        Logger.log_scalar("dqn_agent/target_q_value/min", target_q_values.min().item())
        Logger.log_scalar(
            "dqn_agent/target_q_value/mean", target_q_values.mean().item()
        )
        Logger.log_scalar("dqn_per_agent/sampling_weights/max", sampling_weights.max())
        Logger.log_scalar("dqn_per_agent/sampling_weights/min", sampling_weights.min())
        Logger.log_scalar(
            "dqn_per_agent/sampling_weights/mean", sampling_weights.mean()
        )

    def _compute_td_error(
        self, transition: Transition
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Compute the TD error for a transition or batch of transitions.

        Args:
            transition (Transition): The transition.

        Returns:
            tuple[torch.Tensor, torch.Tensor, torch.Tensor]: The TD error, current Q-values and
                target Q-values.
        """
        # Unpack the transition
        observation, action, reward, next_observation, done = transition[
            self._relevant_keys
        ]

        # Convert the numpy arrays to torch tensors
        observation = torch.tensor(observation, dtype=torch.float32)
        action = torch.tensor(action, dtype=torch.long)
        reward = torch.tensor(reward, dtype=torch.float32)
        next_observation = torch.tensor(next_observation, dtype=torch.float32)
        done = torch.tensor(done, dtype=torch.float32)

        # Compute the target Q-value
        target_q_values = (
            self._model.predict(next_observation, target=True).max(dim=-1).values
        )
        target_q_values = reward.squeeze(dim=-1) + self._gamma * target_q_values * (
            1 - done.squeeze(dim=-1)
        )

        # Compute the current Q-value
        current_q_values = self._model.predict(observation)
        current_q_values = current_q_values.gather(dim=-1, index=action).squeeze(dim=-1)

        # Compute TD error
        return target_q_values - current_q_values, current_q_values, target_q_values

    def train(self) -> None:
        """Set the agent to training mode."""
        self._model.train()
        self._epsilon.train()

    def test(self) -> None:
        """Set the agent to testing mode."""
        self._model.test()
        self._epsilon.test()
