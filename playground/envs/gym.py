import numpy as np
import gymnasium as gym
from PIL import Image

from playground.envs.common import Environment, TimeStep, StepType


class GymEnvironment(Environment):
    def __init__(self, env: gym.Env):
        self._env = env

    def reset(self) -> TimeStep:
        observation, _ = self._env.reset()

        action = None
        if isinstance(self._env.action_space, gym.spaces.Discrete):
            action = np.array(-1, dtype=np.int32)
        elif isinstance(self._env.action_space, gym.spaces.Box):
            action = np.zeros(self._env.action_space.shape, dtype=np.float32)

        reward = np.array(0, dtype=np.float32)
        return TimeStep(
            step_type=StepType.FIRST, truncated=False, observation=observation, action=action, reward=reward
        )

    def step(self, action: np.ndarray) -> TimeStep:
        observation, reward, terminated, truncated, _ = self._env.step(action)

        step_type = StepType.MID if not (terminated or truncated) else StepType.LAST
        reward = np.array(reward, dtype=np.float32)

        return TimeStep(
            step_type=step_type,
            truncated=truncated,
            observation=observation,
            action=action,
            reward=reward,
        )

    def observation_space(self) -> gym.Space:
        return self._env.observation_space

    def action_space(self) -> gym.Space:
        return self._env.action_space

    def render(self, width: int, height: int) -> Image:
        return Image.fromarray(self._env.render()).resize((width, height))


def make_env(task="LunarLander-v3"):
    env = gym.make(task, render_mode="rgb_array")
    env = GymEnvironment(env)
    return env
