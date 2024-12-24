import abc

import numpy as np
import gymnasium as gym
from PIL import Image

from playground.envs.common import Environment, TimeStep, StepType


class GymEnvironment(Environment, abc.ABC):
    def __init__(self, env: gym.Env):
        self._env = env

    def render(self, width: int, height: int) -> Image:
        return Image.fromarray(self._env.render()).resize((width, height))


class DiscreteGymEnvironment(GymEnvironment):
    def __init__(self, env: gym.Env):
        super().__init__(env)

    def reset(self) -> TimeStep:
        observation, _ = self._env.reset()
        reward = np.array(0, dtype=np.float32)
        return TimeStep(step_type=StepType.FIRST, truncated=False, observation=observation, action=None, reward=reward)

    def step(self, action: np.ndarray) -> TimeStep:
        f_action = action + self._env.action_space.start
        observation, reward, terminated, truncated, _ = self._env.step(f_action)
        reward = np.array(reward, dtype=np.float32)
        return TimeStep(
            step_type=StepType.FIRST, truncated=False, observation=observation, action=action, reward=reward
        )

    @property
    def observation_space(self) -> gym.spaces.Box:
        return self._env.observation_space

    @property
    def action_space(self) -> gym.spaces.Discrete:
        return gym.spaces.Discrete(n=self._env.action_space.n, start=0)


class ContinuousGymEnvironment(GymEnvironment):
    def __init__(self, env: gym.Env):
        super().__init__(env)

    def reset(self) -> TimeStep:
        observation, _ = self._env.reset()
        reward = np.array(0, dtype=np.float32)
        return TimeStep(step_type=StepType.FIRST, truncated=False, observation=observation, action=None, reward=reward)

    def step(self, action: np.ndarray) -> TimeStep:
        f_action = self._env.action_space.low + (action - self.action_space.low) * (
            (self._env.action_space.high - self._env.action_space.low)
            / (self.action_space.high - self.action_space.low)
        )
        observation, reward, terminated, truncated, _ = self._env.step(f_action)
        reward = np.array(reward, dtype=np.float32)
        return TimeStep(
            step_type=StepType.FIRST, truncated=False, observation=observation, action=action, reward=reward
        )

    @property
    def observation_space(self) -> gym.spaces.Box:
        return self._env.observation_space

    @property
    def action_space(self) -> gym.spaces.Box:
        return gym.spaces.Box(
            low=-np.ones(self._env.action_space.shape, dtype=np.float32),
            high=np.ones(self._env.action_space.shape, dtype=np.float32),
            shape=self._env.action_space.shape,
            dtype=np.float32,
        )


def make_env(task="LunarLander-v3"):
    env = gym.make(task, render_mode="rgb_array")

    if not isinstance(env.observation_space, gym.spaces.Box):
        raise TypeError(f"[{env.observation_space.__class__.__name__}] observation space not supported]")

    if isinstance(env.action_space, gym.spaces.Discrete):
        env = DiscreteGymEnvironment(env)
    elif isinstance(env.action_space, gym.spaces.Box):
        env = ContinuousGymEnvironment(env)
    else:
        raise TypeError(f"[{env.action_space.__class__.__name__}] action space not supported")
    return env
