import abc
import enum
from typing import NamedTuple
import numpy as np
import gymnasium as gym
from PIL import Image


class StepType(enum.IntEnum):
    FIRST = 0
    MID = 1
    LAST = 2


class TimeStep(NamedTuple):
    step_type: StepType
    truncated: bool
    observation: np.ndarray
    action: np.ndarray | None
    reward: np.ndarray

    def first(self):
        return self.step_type == StepType.FIRST

    def mid(self):
        return self.step_type == StepType.MID

    def last(self):
        return self.step_type == StepType.LAST

    def __getitem__(self, attr):
        return getattr(self, attr)


class Environment(metaclass=abc.ABCMeta):
    def __init__(self):
        self.episode_length = None

    @abc.abstractmethod
    def reset(self) -> TimeStep:
        pass

    @abc.abstractmethod
    def step(self, action: np.ndarray) -> TimeStep:
        pass

    @property
    @abc.abstractmethod
    def observation_space(self) -> gym.spaces.Box:
        pass

    @property
    @abc.abstractmethod
    def action_space(self) -> gym.Space:
        pass

    @abc.abstractmethod
    def render(self, im_w: int, im_h: int) -> Image:
        pass
