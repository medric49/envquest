import abc
import gymnasium as gym
import numpy as np
import torch

from playground.envs.common import TimeStep


def init_weights(m):
    if isinstance(m, torch.nn.Linear):
        torch.nn.init.xavier_uniform(m.weight)
        m.bias.data.fill_(0.0)


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
    def act(self, observation: np.ndarray, **kwargs) -> np.ndarray:
        pass

    @abc.abstractmethod
    def improve(self, **kwargs) -> dict:
        pass


class OneActionAgent(Agent):
    def __init__(self, action: np.ndarray, **kwargs):
        super().__init__(**kwargs)
        self._action = action

    def memorize(self, timestep: TimeStep, next_timestep: TimeStep):
        pass

    def act(self, observation: np.ndarray, **kwargs) -> np.ndarray:
        return self._action

    def improve(self, **kwargs) -> dict:
        return {}


class RandomAgent(Agent):

    def memorize(self, timestep: TimeStep, next_timestep: TimeStep):
        pass

    def act(self, *args) -> np.ndarray:
        return self.action_space.sample()

    def improve(self, **kwargs) -> dict:
        return {}
