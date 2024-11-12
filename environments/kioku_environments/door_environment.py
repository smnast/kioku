"""
door_environment.py

Provides a reinforcement learning environment where an agent must choose the correct door after a series of steps.
"""

import numpy as np
from environments.environment import Environment


class DoorEnvironment(Environment):
    """
    DoorEnvironment is a reinforcement learning environment where an agent must choose the correct door 
    after a series of steps. The environment provides observations, rewards, and handles the episode 
    termination logic.

    Attributes:
        action_size (int): The number of actions that can be taken.
        observation_size (int): The dimension of the observation space.
        continuous (bool): Whether the environment has a continuous action space.
        _num_doors (int): Number of doors in the environment.
        _flash_steps (int): Number of steps during which the correct door is flashed.
        _reward_delay (int): Delay in steps before the agent can receive a reward for choosing the correct door.
        _print_observation (bool): Flag to print the observation at each step.
        _max_steps (int): Maximum number of steps in an episode.
        _chosen_door (int): The door chosen by the environment.
        _step (int): Current step in the episode.
    """
    def __init__(
        self,
        num_doors: int = 10,
        flash_steps: int = 10,
        reward_delay: int = 10,
        print_observation: bool = False,
    ) -> None:
        self._num_doors = num_doors
        self._flash_steps = flash_steps
        self._reward_delay = reward_delay
        self._print_observation = print_observation
        self._max_steps = 30

    def reset(self) -> np.ndarray:
        self._chosen_door = np.random.randint(0, self._num_doors)
        observation = np.zeros(self._num_doors, dtype=np.float32)
        observation[self._chosen_door] = 1

        # Print the observation
        if self._print_observation:
            print(f'observation: {observation}')

        self._step = 0

        return observation

    def step(self, action: np.ndarray) -> tuple[np.ndarray, ...]:
        # Increase step counter
        self._step += 1

        # Update the observation
        observation = np.zeros(self._num_doors, dtype=np.float32)
        if self._step < self._flash_steps:
            observation[self._chosen_door] = 1

        # Print the observation
        if self._print_observation:
            print(f'observation: {observation}, action: {action}')

        # Give the agent a positive reward for choosing the correct door after the delay
        # Otherwise, if the agent acted, give a negative reward
        reward = np.array([0], dtype=np.float32)
        chosen_door = action.item()
        if chosen_door != 0:
            if (
                chosen_door == self._chosen_door + 1
                and self._step >= self._flash_steps + self._reward_delay
            ):
                reward += 1
            else:
                reward -= 1

        # Truncate the episode if the maximum number of steps is reached
        done = np.array([0], dtype=bool)
        truncated = np.array([int(self._step >= self._max_steps)], dtype=bool)

        return observation, reward, done, truncated

    @property
    def action_size(self) -> int:
        return self._num_doors + 1

    @property
    def observation_size(self) -> int:
        return self._num_doors

    @property
    def continuous(self) -> bool:
        return False
