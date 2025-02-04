import math

import gymnasium as gym
import numpy as np
import torch
from torch import distributions

from envquest import utils
from envquest.agents.common import Agent
from envquest.envs.common import TimeStep
from envquest.functions.policies import DiscretePolicyNet
from envquest.memories.replay_memories import ReplayMemory


class DiscreteREINFORCEAgent(Agent):
    def __init__(
        self,
        mem_capacity: int,
        discount: float,
        lr: float,
        observation_space: gym.spaces.Box,
        action_space: gym.spaces.Discrete,
    ):
        super().__init__(observation_space=observation_space, action_space=action_space)

        self.memory = ReplayMemory(mem_capacity, discount, n_steps=math.inf)
        self.policy = DiscretePolicyNet(observation_space.shape[0], action_space.n).to(device=utils.device())
        self.policy.apply(utils.init_weights)
        self.optimizer = torch.optim.Adam(self.policy.parameters(), lr=lr)

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

    def improve(self, **kwargs) -> dict:
        if len(self.memory) == 0:
            return {}

        obs, action, _, rtg, _, _ = self.memory.sample(size=self.policy_batch_size, recent=True)

        obs = torch.tensor(obs, dtype=torch.float32, device=utils.device())
        action = torch.tensor(action, dtype=torch.int64, device=utils.device())
        stand_rtg = utils.standardize(rtg, rtg.mean(), rtg.std())
        stand_rtg = torch.tensor(stand_rtg, dtype=torch.float32, device=utils.device())

        self.policy.train()
        self.optimizer.zero_grad()
        pred_action = self.policy(obs)
        pred_action_dist = distributions.Categorical(pred_action)
        loss = -pred_action_dist.log_prob(action) * stand_rtg
        loss = loss.mean()
        loss.backward()
        self.optimizer.step()

        self.last_policy_improvement_step = self.step_count
        self.memory.initialize()

        return {
            "train/batch/reward": rtg.mean().item(),
            "train/batch/loss": loss.item(),
            "train/batch/entropy": pred_action_dist.entropy().mean().item(),
        }
