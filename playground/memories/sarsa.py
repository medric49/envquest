from collections import deque
import numpy as np

from playground.envs.common import TimeStep
from playground.memories.common import AgentMemory


class SarsaAgentMemory(AgentMemory):
    def __init__(self, capacity: int, discount: float):
        super().__init__(capacity, discount)

        self.observations = None
        self.actions = None
        self.rewards = None
        self.next_step_terminal = None

        self.initialize()

    def initialize(self):
        self.observations = deque(maxlen=self.capacity)
        self.actions = deque(maxlen=self.capacity)
        self.rewards = deque(maxlen=self.capacity)
        self.next_step_terminal = deque(maxlen=self.capacity)

    def push(self, timestep: TimeStep, next_timestep: TimeStep):
        observation = timestep.observation
        action = next_timestep.action
        reward = next_timestep.reward
        next_step_terminal = next_timestep.last()

        self.observations.append(observation)
        self.actions.append(action)
        self.rewards.append(reward)
        self.next_step_terminal.append(next_step_terminal)

        if next_timestep.last():
            self.observations.append(np.zeros_like(observation))
            self.actions.append(np.zeros_like(action))

    def __len__(self):
        return len(self.observations) - 1

    def sample(self, size: int, **kwargs) -> tuple[np.ndarray, ...]:
        indices = np.arange(len(self), dtype=np.int32)
        indices = indices[-size:]

        observations = np.stack(self.observations)[indices]
        actions = np.stack(self.actions)[indices]
        rewards = np.stack(self.rewards)[indices]
        next_observations = np.stack(self.observations)[indices + 1]
        next_actions = np.stack(self.actions)[indices + 1]
        next_step_terminal = np.array(self.next_step_terminal, dtype=np.uint8)[indices]

        return (
            observations,
            actions,
            rewards,
            next_observations,
            next_actions,
            next_step_terminal,
        )
