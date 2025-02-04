import gymnasium as gym
import torch
from torch import distributions

from envquest import utils
from envquest.agents.pg_agents import DiscretePGAgent


class DiscretePPOAgent(DiscretePGAgent):
    def __init__(
        self,
        mem_capacity: int,
        discount: float,
        lr: float,
        clip_eps: float,
        num_policy_updates: int,
        observation_space: gym.spaces.Box,
        action_space: gym.spaces.Discrete,
    ):
        super().__init__(mem_capacity, discount, lr, observation_space, action_space)
        self.clip_eps = clip_eps
        self.num_policy_updates = num_policy_updates

    def improve_actor(self) -> dict:
        obs, action, reward, rtg, _, _ = self.memory.sample(size=self.policy_batch_size, recent=True)

        obs = torch.tensor(obs, dtype=torch.float32, device=utils.device())
        action = torch.tensor(action, dtype=torch.int64, device=utils.device())
        rtg = torch.tensor(rtg, dtype=torch.float32, device=utils.device())

        self.v_net.eval()
        self.policy.eval()
        with torch.no_grad():
            obs_value = self.v_net(obs).flatten()
            unstand_obs_value = utils.unstandardize(obs_value, self.batch_rtg_mean, self.batch_rtg_std)

            advantage = rtg - unstand_obs_value

            stand_advantage = utils.standardize(advantage, advantage.mean(), advantage.std())

            source_action_probs = self.policy(obs)
            source_action_dist = distributions.Categorical(source_action_probs)
            source_action_prob = source_action_probs.gather(dim=1, index=action.unsqueeze(dim=1)).flatten()

            g = (1 + torch.sign(stand_advantage) * self.clip_eps) * stand_advantage

        self.policy.train()

        for _ in range(self.num_policy_updates):
            self.policy_optimizer.zero_grad()
            action_prob = self.policy(obs).gather(dim=1, index=action.unsqueeze(dim=1)).flatten()
            loss = -torch.stack([(action_prob / source_action_prob) * stand_advantage, g], dim=1).min(dim=1)[0]
            loss = loss.mean()
            loss.backward()
            self.policy_optimizer.step()
            loss = loss.item()

        return {
            "train/batch/p_reward": reward.mean().item(),
            "train/batch/advantage": advantage.mean().item(),
            "train/batch/p_loss": loss,
            "train/batch/entropy": source_action_dist.entropy().mean().item(),
        }
