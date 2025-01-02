import abc

import numpy as np

from envquest.envs.common import TimeStep


class AgentMemory(metaclass=abc.ABCMeta):

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
    def sample(self, **kwargs) -> tuple[np.ndarray, ...]:
        pass
