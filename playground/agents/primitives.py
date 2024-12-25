import numpy as np

from playground.agents.common import Agent
from playground.envs.common import TimeStep


class OneActionAgent(Agent):
    def __init__(self, action: np.ndarray, **kwargs):
        super().__init__(**kwargs)
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
