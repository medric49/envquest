import abc
import math

import gymnasium as gym
import numpy as np
import torch
from torch import distributions

from envquest import utils
from envquest.agents.common import Agent
from envquest.envs.common import TimeStep
from envquest.functions.policies import DiscretePolicyNet, ContinuousPolicyNet
from envquest.functions.v_values import DiscreteVNet
from envquest.memories.replay_memories import ReplayMemory


class PGAgent(Agent, abc.ABC):
    def __init__(
        self,
        mem_capacity: int,
        discount: float,
        lr: float,
        observation_space: gym.spaces.Box,
        action_space: gym.spaces.Space,
    ):
        super().__init__(observation_space, action_space)

        self.memory = ReplayMemory(mem_capacity, discount, n_steps=math.inf)
        self.discount = discount

        self.batch_rtg_mean = None
        self.batch_rtg_std = None

        self.v_net = DiscreteVNet(observation_space.shape[0]).to(device=utils.device())
        self.v_net.apply(utils.init_weights)
        self.v_net_optimizer = torch.optim.Adam(self.v_net.parameters(), lr=lr)
        self.criterion = torch.nn.MSELoss()

        self.last_policy_improvement_step = 0
        self.step_count = 0

    def memorize(self, timestep: TimeStep, next_timestep: TimeStep):
        self.step_count += 1
        self.memory.push(timestep, next_timestep)

    @property
    def policy_batch_size(self):
        return self.step_count - self.last_policy_improvement_step

    def improve(self, batch_size=None, **kwargs) -> dict:
        # if batch_size is None:
        #     raise ValueError("'batch_size' is required")
        if len(self.memory) == 0:
            return {}

        metrics = {}
        if self.batch_rtg_std is not None and self.batch_rtg_mean is not None:
            metrics.update(self.improve_actor())
        metrics.update(self.improve_critic())

        self.last_policy_improvement_step = self.step_count
        self.memory.initialize()
        return metrics

    def improve_critic(self) -> dict:
        obs, _, _, rtg, _, _ = self.memory.sample(size=self.policy_batch_size, recent=True)

        self.batch_rtg_mean = rtg.mean()
        self.batch_rtg_std = rtg.std()

        obs = torch.tensor(obs, dtype=torch.float32, device=utils.device())
        stand_rtg = utils.standardize(rtg, self.batch_rtg_mean, self.batch_rtg_std)
        stand_rtg = torch.tensor(stand_rtg, dtype=torch.float32, device=utils.device())

        self.v_net.train()
        self.v_net_optimizer.zero_grad()
        obs_value = self.v_net(obs).flatten()
        loss = self.criterion(obs_value, stand_rtg)
        loss.backward()
        self.v_net_optimizer.step()

        unstand_obs_value = utils.unstandardize(
            obs_value.cpu().detach().numpy(), self.batch_rtg_mean, self.batch_rtg_std
        )
        return {
            "train/batch/v_rtg": rtg.mean(),
            "train/batch/v_loss": loss.item(),
            "train/batch/v_value": unstand_obs_value.mean(),
        }

    @abc.abstractmethod
    def improve_actor(self) -> dict:
        pass


class DiscretePGAgent(PGAgent, abc.ABC):
    def __init__(
        self,
        mem_capacity: int,
        discount: float,
        lr: float,
        observation_space: gym.spaces.Box,
        action_space: gym.spaces.Discrete,
    ):
        super().__init__(mem_capacity, discount, lr, observation_space, action_space)

        self.policy = DiscretePolicyNet(observation_space.shape[0], action_space.n).to(device=utils.device())
        self.policy.apply(utils.init_weights)
        self.policy_optimizer = torch.optim.Adam(self.policy.parameters(), lr=lr)

    def act(self, observation: np.ndarray = None, noisy=False, **kwargs) -> np.ndarray:
        observation = torch.tensor(observation, dtype=torch.float32, device=utils.device())
        observation = torch.unsqueeze(observation, dim=0)

        self.policy.eval()
        with torch.no_grad():
            action = self.policy(observation).flatten()

            if not noisy:
                action = action.argmax().item()
                action = np.asarray(action, dtype=np.int64)
            else:
                action_dist = distributions.Categorical(action)
                action = action_dist.sample().item()
                action = np.asarray(action, dtype=np.int64)
        return action


class ContinuousPGAgent(PGAgent, abc.ABC):
    def __init__(
        self,
        mem_capacity: int,
        discount: float,
        lr: float,
        observation_space: gym.spaces.Box,
        action_space: gym.spaces.Box,
    ):
        super().__init__(mem_capacity, discount, lr, observation_space, action_space)

        self.policy = ContinuousPolicyNet(observation_space.shape[0], action_space.shape[0]).to(device=utils.device())
        self.policy.apply(utils.init_weights)
        self.policy_optimizer = torch.optim.Adam(self.policy.parameters(), lr=lr)

        self.noise = 0.1

    def act(self, observation: np.ndarray = None, noisy=False, **kwargs) -> np.ndarray:
        observation = torch.tensor(observation, dtype=torch.float32, device=utils.device())
        observation = torch.unsqueeze(observation, dim=0)

        self.policy.eval()
        with torch.no_grad():
            action = self.policy(observation).flatten()

            if not noisy:
                action = action.cpu().numpy()
                return action
            else:
                action_dist = distributions.Normal(action, self.noise)
                action = action_dist.sample().cpu().numpy()
        return action
