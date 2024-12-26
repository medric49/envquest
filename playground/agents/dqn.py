import gymnasium as gym
import numpy as np
import torch
from torch import nn

from playground import utils
from playground.agents.common import Agent
from playground.utils import init_weights
from playground.envs.common import TimeStep
from playground.memories import ReplayMemory


class DiscreteQNet(nn.Module):

    def __init__(self, observation_dim: int, num_actions: int):
        super().__init__()
        self.layer1 = nn.Linear(observation_dim, 128)
        self.layer2 = nn.Linear(128, 128)
        self.layer3 = nn.Linear(128, num_actions)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.relu(self.layer1(x))
        x = self.relu(self.layer2(x))
        return self.layer3(x)


class DiscreteQNetAgent(Agent):
    def __init__(
        self,
        mem_capacity: int,
        discount: float,
        lr: float,
        tau: float,
        eps_start: float,
        eps_end: float,
        eps_step_duration: float,
        observation_space: gym.spaces.Box,
        action_space: gym.spaces.Discrete,
    ):
        super().__init__(observation_space=observation_space, action_space=action_space)

        self.memory = ReplayMemory(mem_capacity, discount)
        observation_dim = observation_space.shape[0]

        self.q_net = DiscreteQNet(observation_dim, action_space.n).to(device=utils.device())
        self.target_q_net = DiscreteQNet(observation_dim, action_space.n).to(device=utils.device())
        self.q_net.apply(init_weights)
        self.target_q_net.load_state_dict(self.q_net.state_dict())
        self.optimizer = torch.optim.Adam(self.q_net.parameters(), lr=lr)
        self.criterion = torch.nn.MSELoss()
        self.discount = discount
        self.tau = tau

        self.step_count = 0
        self.eps_start = eps_start
        self.eps_end = eps_end
        self.eps_step_duration = eps_step_duration

    @property
    def current_noise(self):
        # mix = np.clip(self.step_count / self.eps_step_duration, 0.0, 1.0)
        mix = 1 - np.exp(-4 * self.step_count / self.eps_step_duration)
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

    def improve(self, batch_size: int = None, **kwargs) -> dict:
        if batch_size is None:
            raise ValueError("batch_size is required")

        if len(self.memory) == 0:
            return {}

        obs, action, reward, next_obs, next_obs_terminal = self.memory.get_steps(size=batch_size)
        obs = torch.tensor(obs, dtype=torch.float32, device=utils.device())
        next_obs = torch.tensor(next_obs, dtype=torch.float32, device=utils.device())

        action = torch.tensor(action, dtype=torch.int64, device=utils.device())
        reward = torch.tensor(reward, dtype=torch.float32, device=utils.device())
        next_obs_terminal = torch.tensor(next_obs_terminal, dtype=torch.float32, device=utils.device())

        self.q_net.eval()
        self.target_q_net.eval()
        with torch.no_grad():
            next_action = self.q_net(next_obs).max(dim=1)[1].to(dtype=torch.int64)
            next_value = (self.target_q_net(next_obs).gather(dim=1, index=next_action.unsqueeze(dim=1)).flatten()) * (
                1.0 - next_obs_terminal
            )
        target = reward + self.discount * next_value

        self.q_net.train()
        self.optimizer.zero_grad()
        value = self.q_net(obs)
        value = value.gather(dim=1, index=action.unsqueeze(dim=1))
        value = value.flatten()
        loss = self.criterion(value, target)
        loss.backward()
        self.optimizer.step()

        target_state_dict = self.target_q_net.state_dict()
        source_state_dict = self.q_net.state_dict()
        for key in source_state_dict:
            target_state_dict[key] = source_state_dict[key] * self.tau + target_state_dict[key] * (1 - self.tau)
        self.target_q_net.load_state_dict(target_state_dict)

        return {
            "train/batch/reward": reward.mean().item(),
            "train/batch/q_value": value.mean().item(),
            "train/batch/q_value_loss": loss.item(),
            "train/batch/next_value": next_value.mean().item(),
        }