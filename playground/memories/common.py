import abc

import numpy as np

from playground.envs.common import TimeStep


class AgentMemory(metaclass=abc.ABCMeta):
    def __init__(self, capacity: int, discount: float):
        self.discount = discount
        self.capacity = capacity

    @abc.abstractmethod
    def initialize(self):
        pass

    @abc.abstractmethod
    def push(self, timestep: TimeStep, next_timestep: TimeStep):
        pass

    @abc.abstractmethod
    def __len__(self):
        pass

    @abc.abstractmethod
    def sample(self, size: int, **kwargs) -> tuple[np.ndarray, ...]:
        pass
