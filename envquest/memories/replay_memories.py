from collections import deque
from typing import Union

import numpy as np

from envquest.envs.common import TimeStep
from envquest.memories.common import AgentMemory


class ReplayMemory(AgentMemory):
    def __init__(self, capacity: int, discount: float, n_steps: Union[int, float]):
        self.discount = discount
        self.capacity = capacity
        self.n_steps = n_steps
        self._max_offset = 0

        self.observations = None
        self.actions = None
        self.rewards = None
        self.n_steps_rewards = None
        self.next_observations = None
        self.next_step_terminal = None

        self.initialize()

    def initialize(self):
        self.observations = deque(maxlen=self.capacity)
        self.actions = deque(maxlen=self.capacity)
        self.rewards = deque(maxlen=self.capacity)
        self.n_steps_rewards = deque(maxlen=self.capacity)
        self.next_observations = deque(maxlen=self.capacity)
        self.next_step_terminal = deque(maxlen=self.capacity)

    def push(self, timestep: TimeStep, next_timestep: TimeStep):
        if timestep.first():
            self._max_offset = 0
        else:
            self._max_offset = min(self._max_offset + 1, self.n_steps - 1)
        observation = timestep.observation
        action = next_timestep.action
        reward = next_timestep.reward
        next_observation = next_timestep.observation
        next_step_terminal = next_timestep.last() and not next_timestep.truncated

        self.observations.append(observation)
        self.actions.append(action)
        self.rewards.append(reward)
        self.n_steps_rewards.append(reward)
        self.next_observations.append(next_observation)
        self.next_step_terminal.append(next_step_terminal)

        for offset in range(self._max_offset, 0, -1):
            index = len(self.n_steps_rewards) - offset
            self.n_steps_rewards[index] += (self.discount**offset) * reward

    def __len__(self):
        return len(self.observations)

    def sample(self, size: int = None, recent: bool = False, **kwargs) -> tuple[np.ndarray, ...]:
        if size is None:
            raise ValueError("'size' is required")

        indices = np.arange(len(self), dtype=np.int32)
        if not recent:
            indices = np.random.choice(indices, size=indices.shape[0], replace=False)
        indices = indices[-size:]

        observations = np.stack(self.observations)[indices]
        actions = np.stack(self.actions)[indices]
        next_observations = np.stack(self.next_observations)[indices]
        next_step_terminal = np.array(self.next_step_terminal, dtype=np.uint8)[indices]
        rewards = np.stack(self.rewards)[indices]
        n_steps_rewards = np.stack(self.n_steps_rewards)[indices]

        return (
            observations,
            actions,
            rewards,
            n_steps_rewards,
            next_observations,
            next_step_terminal,
        )


class SarsaAgentMemory(AgentMemory):
    def __init__(self):
        self.observations = None
        self.actions = None
        self.rewards = None
        self.next_observations = None
        self.next_actions = None
        self.next_step_terminal = None

        self.max_len = 2

        self.initialize()

    def initialize(self):
        self.observations = deque(maxlen=self.max_len)
        self.actions = deque(maxlen=self.max_len)
        self.rewards = deque(maxlen=self.max_len)
        self.next_step_terminal = deque(maxlen=self.max_len)

    def push(self, timestep: TimeStep, next_timestep: TimeStep):
        observation = timestep.observation  # s_t
        action = next_timestep.action  # a_t
        reward = next_timestep.reward  # r_{t+1}
        next_step_terminal = next_timestep.last()

        self.observations.append(observation)
        self.actions.append(action)
        self.rewards.append(reward)
        self.next_step_terminal.append(next_step_terminal)

    def __len__(self):
        if len(self.observations) == 0:
            return 0
        return len(self.observations) - 1

    def sample(self, **kwargs) -> tuple[np.ndarray, ...]:
        observations = np.stack(self.observations)[:-1]
        actions = np.stack(self.actions)[:-1]
        rewards = np.stack(self.rewards)[:-1]
        next_observations = np.stack(self.observations)[1:]
        next_actions = np.stack(self.actions)[1:]
        next_step_terminal = np.array(self.next_step_terminal, dtype=np.uint8)[:-1]

        return (
            observations,
            actions,
            rewards,
            next_observations,
            next_actions,
            next_step_terminal,
        )
