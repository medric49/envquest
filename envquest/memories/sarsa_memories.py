from collections import deque
import numpy as np

from envquest.envs.common import TimeStep
from envquest.memories.common import AgentMemory


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
