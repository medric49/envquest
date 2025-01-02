import numpy as np

import gymnasium as gym

from envquest.agents.common import Agent
from envquest.envs.common import TimeStep


class OneActionAgent(Agent):
    def __init__(self, action: np.ndarray, observation_space: gym.spaces.Space, action_space: gym.spaces.Space):
        super().__init__(observation_space, action_space)
        self._action = action

    def memorize(self, timestep: TimeStep, next_timestep: TimeStep):
        pass

    def act(self, **kwargs) -> np.ndarray:
        return self._action

    def improve(self, **kwargs) -> dict:
        return {}


class RandomAgent(Agent):

    def memorize(self, timestep: TimeStep, next_timestep: TimeStep):
        pass

    def act(self, **kwargs) -> np.ndarray:
        return self.action_space.sample()

    def improve(self, **kwargs) -> dict:
        return {}
