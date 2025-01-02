import gymnasium as gym
import numpy as np
from PIL import Image

from envquest.envs.common import Wrapper, TimeStep, StepType


class MaxEpisodeLengthWrapper(Wrapper):
    def __init__(self, env, max_episode_length):
        super().__init__(env)
        self.max_episode_length = max_episode_length

    def reset(self) -> TimeStep:
        return self._env.reset()

    def step(self, action: np.ndarray) -> TimeStep:
        timestep = self._env.step(action)
        if self.episode_length >= self.max_episode_length and not timestep.last():
            timestep = timestep._replace(step_type=StepType.LAST, truncated=True)
        return timestep

    @property
    def observation_space(self) -> gym.spaces.Box:
        return self._env.observation_space

    @property
    def action_space(self) -> gym.Space:
        return self._env.action_space

    def render(self, im_w: int, im_h: int) -> Image:
        return self._env.render(im_w, im_h)
