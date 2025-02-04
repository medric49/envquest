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


class DiscreteVanillaPGAgent(DiscretePGAgent):
    def improve_actor(self) -> dict:
        obs, action, reward, _, next_obs, next_obs_terminal = self.memory.sample(
            size=self.policy_batch_size, recent=True
        )

        obs = torch.tensor(obs, dtype=torch.float32, device=utils.device())
        action = torch.tensor(action, dtype=torch.int64, device=utils.device())
        reward = torch.tensor(reward, dtype=torch.float32, device=utils.device())
        next_obs = torch.tensor(next_obs, dtype=torch.float32, device=utils.device())
        next_obs_terminal = torch.tensor(next_obs_terminal, dtype=torch.float32, device=utils.device())

        self.v_net.eval()
        with torch.no_grad():
            obs_value = self.v_net(obs).flatten()
            unstand_obs_value = utils.unstandardize(obs_value, self.batch_rtg_mean, self.batch_rtg_std)

            next_obs_value = self.v_net(next_obs).flatten()
            unstand_next_obs_value = utils.unstandardize(next_obs_value, self.batch_rtg_mean, self.batch_rtg_std)

            advantage = reward + self.discount * unstand_next_obs_value * (1 - next_obs_terminal) - unstand_obs_value

        stand_advantage = utils.standardize(advantage, advantage.mean(), advantage.std())

        self.policy.train()
        self.policy_optimizer.zero_grad()
        pred_action = self.policy(obs)
        pred_action_dist = distributions.Categorical(pred_action)
        loss = -pred_action_dist.log_prob(action) * stand_advantage
        loss = loss.mean()
        loss.backward()
        self.policy_optimizer.step()

        return {
            "train/batch/p_reward": reward.mean().item(),
            "train/batch/advantage": advantage.mean().item(),
            "train/batch/p_loss": loss.item(),
            "train/batch/entropy": pred_action_dist.entropy().mean().item(),
        }


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


class ContinuousVanillaPGAgent(ContinuousPGAgent):
    def improve_actor(self) -> dict:
        obs, action, reward, rtg, _, _ = self.memory.sample(size=self.policy_batch_size, recent=True)

        obs = torch.tensor(obs, dtype=torch.float32, device=utils.device())
        action = torch.tensor(action, dtype=torch.float32, device=utils.device())
        rtg = torch.tensor(rtg, dtype=torch.float32, device=utils.device())

        self.v_net.eval()
        with torch.no_grad():
            obs_value = self.v_net(obs).flatten()
            advantage = rtg - obs_value

        stand_advantage = utils.standardize(advantage, advantage.mean(), advantage.std())

        self.policy.train()
        self.policy_optimizer.zero_grad()
        pred_action = self.policy(obs)
        pred_action_dist = distributions.Normal(pred_action, self.noise)
        log_prob = pred_action_dist.log_prob(action).sum(dim=-1).flatten()
        loss = -log_prob * stand_advantage
        loss = loss.mean()
        loss.backward()
        self.policy_optimizer.step()

        return {
            "train/batch/p_reward": rtg.mean().item(),
            "train/batch/advantage": advantage.mean().item(),
            "train/batch/p_loss": loss.item(),
            "train/batch/entropy": pred_action_dist.entropy().mean().item(),
        }
