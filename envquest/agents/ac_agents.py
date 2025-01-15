import numpy as np
import torch

import gymnasium as gym
from torch import distributions

from envquest import utils
from envquest.agents.common import Agent
from envquest.envs.common import TimeStep
from envquest.functions.policies import DiscretePolicyNet
from envquest.functions.v_values import DiscreteVNet
from envquest.memories.replay_memories import ReplayMemory


class DiscreteA2CAgent(Agent):
    def __init__(
        self,
        mem_capacity: int,
        discount: float,
        lr: float,
        observation_space: gym.spaces.Box,
        action_space: gym.spaces.Discrete,
    ):
        super().__init__(observation_space, action_space)

        self.memory = ReplayMemory(mem_capacity, discount, n_steps=1)

        self.policy = DiscretePolicyNet(observation_space.shape[0], action_space.n).to(device=utils.device())
        self.policy.apply(utils.init_weights)
        self.policy_optimizer = torch.optim.Adam(self.policy.parameters(), lr=lr)

        self.v_net = DiscreteVNet(observation_space.shape[0]).to(device=utils.device())
        self.v_net.apply(utils.init_weights)
        self.v_optimizer = torch.optim.Adam(self.v_net.parameters(), lr=lr)

        self.discount = discount
        self.criterion = torch.nn.MSELoss()

        self.last_policy_improvement_step = 0
        self.step_count = 0

    def memorize(self, timestep: TimeStep, next_timestep: TimeStep):
        self.step_count += 1
        self.memory.push(timestep, next_timestep)

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

    @property
    def policy_batch_size(self):
        return self.step_count - self.last_policy_improvement_step

    def improve(self, batch_size: int = None, **kwargs) -> dict:
        if batch_size is None:
            raise ValueError("'batch_size' is required")
        if len(self.memory) == 0:
            return {}

        metrics = {}
        metrics.update(self.improve_policy())
        metrics.update(self.improve_v_net(batch_size))

        return metrics

    def improve_policy(self) -> dict:
        obs, action, reward, next_obs, next_obs_terminal = self.memory.sample(size=self.policy_batch_size, recent=True)

        obs = torch.tensor(obs, dtype=torch.float32, device=utils.device())
        action = torch.tensor(action, dtype=torch.int64, device=utils.device())
        reward = torch.tensor(reward, dtype=torch.float32, device=utils.device())
        next_obs = torch.tensor(next_obs, dtype=torch.float32, device=utils.device())
        next_obs_terminal = torch.tensor(next_obs_terminal, dtype=torch.float32, device=utils.device())

        self.v_net.eval()
        with torch.no_grad():
            obs_value = self.v_net(obs).flatten()
            next_obs_value = self.v_net(next_obs).flatten() * (1 - next_obs_terminal)
            advantage = reward + self.discount * next_obs_value - obs_value
        stand_advantage = utils.standardize(advantage, advantage.mean(), advantage.std())

        self.policy.train()
        self.policy_optimizer.zero_grad()
        pred_action = self.policy(obs)
        pred_action_dist = distributions.Categorical(pred_action)
        loss = -pred_action_dist.log_prob(action) * stand_advantage
        loss = loss.mean()
        loss.backward()
        self.policy_optimizer.step()

        self.last_policy_improvement_step = self.step_count

        return {
            "train/batch/p_reward": reward.mean().item(),
            "train/batch/advantage": advantage.mean().item(),
            "train/batch/p_loss": loss.item(),
            "train/batch/entropy": pred_action_dist.entropy().mean().item(),
        }

    def improve_v_net(self, batch_size) -> dict:
        obs, _, reward, next_obs, next_obs_terminal = self.memory.sample(size=batch_size, recent=False)

        obs = torch.tensor(obs, dtype=torch.float32, device=utils.device())
        reward = torch.tensor(reward, dtype=torch.float32, device=utils.device())
        next_obs = torch.tensor(next_obs, dtype=torch.float32, device=utils.device())
        next_obs_terminal = torch.tensor(next_obs_terminal, dtype=torch.float32, device=utils.device())

        self.v_net.eval()
        with torch.no_grad():
            next_obs_value = self.v_net(next_obs).flatten() * (1 - next_obs_terminal)
            target_value = reward + self.discount * next_obs_value

        self.v_net.train()
        self.v_optimizer.zero_grad()
        obs_value = self.v_net(obs).flatten()
        loss = self.criterion(obs_value, target_value)
        loss.backward()
        self.v_optimizer.step()

        return {
            "train/batch/v_reward": reward.mean().item(),
            "train/batch/v_loss": loss.item(),
            "train/batch/v_value": obs_value.mean().item(),
            "train/batch/v_next_value": next_obs_value.mean().item(),
        }
