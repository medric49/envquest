import numpy as np

import gymnasium as gym
import torch
from torch import distributions

from envquest import utils
from envquest.agents.common import Agent
from envquest.envs.common import TimeStep
from envquest.functions.policies import DiscretePolicyNet
from envquest.memories.online_memories import OnlineMemory


class PGAgent(Agent):
    def __init__(
        self,
        mem_capacity: int,
        discount: float,
        lr: float,
        observation_space: gym.spaces.Box,
        action_space: gym.spaces.Discrete,
    ):
        super().__init__(observation_space=observation_space, action_space=action_space)

        self.memory = OnlineMemory(mem_capacity, discount)

        self.policy = DiscretePolicyNet(observation_space.shape[0], action_space.n).to(device=utils.device())
        self.policy.apply(utils.init_weights)
        self.optimizer = torch.optim.Adam(self.policy.parameters(), lr=lr)

        self.temperature = 1

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
                action_dist = distributions.Categorical(action / self.temperature)
                action = action_dist.sample().item()
                action = np.asarray(action, dtype=np.int64)
        return action

    def improve(self, **kwargs) -> dict:
        if len(self.memory) == 0:
            return {}

        obs, action, reward, _, _ = self.memory.sample(
            size=self.step_count - self.last_policy_improvement_step, recent=True
        )

        obs = torch.tensor(obs, dtype=torch.float32, device=utils.device())
        action = torch.tensor(action, dtype=torch.int64, device=utils.device())
        reward = torch.tensor(reward, dtype=torch.float32, device=utils.device())

        self.policy.train()
        self.optimizer.zero_grad()
        pred_action = self.policy(obs)
        print(pred_action, pred_action.shape)
        pred_action_dist = distributions.Categorical(pred_action / self.temperature)
        loss = -pred_action_dist.log_prob(action)
        print(loss, loss.shape)
        loss = loss.mean()

        loss.backward()
        self.optimizer.step()

        self.last_policy_improvement_step = self.step_count
        return {
            "train/batch/reward": reward.mean().item(),
            "train/batch/loss": loss.item(),
            "train/batch/entropy": distributions.Categorical(pred_action).entropy().mean().item(),
        }
