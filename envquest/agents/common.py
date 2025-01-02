import abc
import gymnasium as gym
import numpy as np

from envquest.envs.common import TimeStep


class EpsilonDecay:
    LINERA = "linear"
    EXPONENTIAL = "exponential"


class Agent(metaclass=abc.ABCMeta):
    def __init__(
        self,
        observation_space: gym.spaces.Space,
        action_space: gym.spaces.Space,
    ):
        self.observation_space = observation_space
        self.action_space = action_space

    @abc.abstractmethod
    def memorize(self, timestep: TimeStep, next_timestep: TimeStep):
        pass

    @abc.abstractmethod
    def act(self, **kwargs) -> np.ndarray:
        pass

    @abc.abstractmethod
    def improve(self, **kwargs) -> dict:
        pass
