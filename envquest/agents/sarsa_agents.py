import gymnasium as gym
import numpy as np
import torch

from envquest import utils
from envquest.agents.common import Agent
from envquest.envs.common import TimeStep
from envquest.functions.q_values import DiscreteQNet
from envquest.memories.replay_memories import SarsaAgentMemory


class DiscreteSarsaAgent(Agent):
    def __init__(
        self,
        discount: float,
        lr: float,
        eps_start: float,
        eps_end: float,
        eps_step_duration: int,
        eps_decay: str,
        observation_space: gym.spaces.Box,
        action_space: gym.spaces.Discrete,
    ):
        super().__init__(observation_space, action_space)

        self.memory = SarsaAgentMemory()
        observation_dim = observation_space.shape[0]

        self.q_net = DiscreteQNet(observation_dim, action_space.n).to(device=utils.device())

        self.optimizer = torch.optim.Adam(self.q_net.parameters(), lr=lr)
        self.criterion = torch.nn.MSELoss()
        self.discount = discount

        self.step_count = 0
        self.eps_start = eps_start
        self.eps_end = eps_end
        self.eps_step_duration = eps_step_duration
        self.eps_decay = eps_decay

    @property
    def current_noise(self):
        if self.eps_decay == "linear":
            mix = np.clip(self.step_count / self.eps_step_duration, 0.0, 1.0)
        elif self.eps_decay == "exponential":
            mix = 1 - np.exp(-4 * self.step_count / self.eps_step_duration)
        else:
            raise ValueError("Invalid value for 'eps_decay'")
        noise = (1.0 - mix) * self.eps_start + mix * self.eps_end
        return noise

    def memorize(self, timestep: TimeStep, next_timestep: TimeStep):
        self.step_count += 1
        self.memory.push(timestep, next_timestep)

    def act(
        self,
        observation: np.ndarray = None,
        random=False,
        noisy=False,
        **kwargs,
    ) -> np.ndarray:

        if random or (noisy and np.random.uniform(0, 1) < self.current_noise):
            return self.action_space.sample()

        observation = torch.tensor(observation, dtype=torch.float32, device=utils.device())
        observation = torch.unsqueeze(observation, dim=0)

        self.q_net.eval()
        with torch.no_grad():
            action = self.q_net(observation).max(dim=1)[1].item()
            action = np.asarray(action, dtype=np.int64)
        return action

    def improve(self, **kwargs) -> dict:

        if len(self.memory) == 0:
            return {}

        obs, action, reward, next_obs, next_action, next_obs_terminal = self.memory.sample()

        obs = torch.tensor(obs, dtype=torch.float32, device=utils.device())
        action = torch.tensor(action, dtype=torch.int64, device=utils.device())
        reward = torch.tensor(reward, dtype=torch.float32, device=utils.device())
        next_obs = torch.tensor(next_obs, dtype=torch.float32, device=utils.device())
        next_action = torch.tensor(next_action, dtype=torch.int64, device=utils.device())
        next_obs_terminal = torch.tensor(next_obs_terminal, dtype=torch.float32, device=utils.device())

        self.q_net.eval()
        with torch.no_grad():
            next_value = self.q_net(next_obs).gather(dim=1, index=next_action.unsqueeze(dim=1)).flatten()
            next_value = next_value * (1.0 - next_obs_terminal)
        target = reward + self.discount * next_value

        self.q_net.train()
        self.optimizer.zero_grad()
        value = self.q_net(obs)
        value = value.gather(dim=1, index=action.unsqueeze(dim=1))
        value = value.flatten()
        loss = self.criterion(value, target)
        loss.backward()
        self.optimizer.step()

        return {
            "train/batch/reward": reward.mean().item(),
            "train/batch/q_value": value.mean().item(),
            "train/batch/q_value_loss": loss.item(),
            "train/batch/next_value": next_value.mean().item(),
            "train/noise": self.current_noise,
        }
