from collections import deque
import numpy as np

from playground.envs.common import TimeStep


class ReplayMemory:
    def __init__(self, capacity: int, discount: float):
        self.discount = discount
        self.capacity = capacity

        self.observations = None
        self.actions = None
        self.rewards = None
        self.next_observations = None
        self.next_terminal_steps = None

        self.initialize()

    def initialize(self):
        self.observations = deque(maxlen=self.capacity)
        self.actions = deque(maxlen=self.capacity)
        self.rewards = deque(maxlen=self.capacity)
        self.next_observations = deque(maxlen=self.capacity)
        self.next_terminal_steps = deque(maxlen=self.capacity)

    def push(self, timestep: TimeStep, next_timestep: TimeStep):
        observation = timestep.observation
        next_observation = next_timestep.observation
        action = next_timestep.action
        reward = next_timestep.reward

        self.observations.append(observation)
        self.actions.append(action)
        self.rewards.append(reward)
        self.next_observations.append(next_observation)
        self.next_terminal_steps.append(next_timestep.last())  # and not next_timestep.truncated

    def __len__(self):
        return len(self.observations)

    def get_steps(self, size: int, recent=False) -> tuple[np.ndarray, ...]:
        indices = np.arange(len(self), dtype=np.int32)
        if not recent:
            indices = np.random.choice(indices, size=indices.shape[0], replace=False)
        indices = indices[-size:]

        observations = np.stack(self.observations)[indices]
        actions = np.stack(self.actions)[indices]
        next_observations = np.stack(self.next_observations)[indices]
        next_terminal_steps = np.array(self.next_terminal_steps, dtype=np.uint8)[indices]
        rewards = np.stack(self.rewards)[indices]

        return (
            observations,
            actions,
            rewards,
            next_observations,
            next_terminal_steps,
        )
