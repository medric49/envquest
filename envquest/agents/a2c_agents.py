import torch
from torch import distributions

from envquest import utils
from envquest.agents.pg_agents import DiscretePGAgent, ContinuousPGAgent


class DiscreteA2CAgent(DiscretePGAgent):
    def improve_actor(self) -> dict:
        obs, action, reward, rtg, _, _ = self.memory.sample(size=self.policy_batch_size, recent=True)

        obs = torch.tensor(obs, dtype=torch.float32, device=utils.device())
        action = torch.tensor(action, dtype=torch.int64, device=utils.device())
        rtg = torch.tensor(rtg, dtype=torch.float32, device=utils.device())

        self.v_net.eval()
        with torch.no_grad():
            obs_value = self.v_net(obs).flatten()
            unstand_obs_value = utils.unstandardize(obs_value, self.batch_rtg_mean, self.batch_rtg_std)

            advantage = rtg - unstand_obs_value

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
            "train/batch/p_reward": reward.mean(),
            "train/batch/advantage": advantage.mean().item(),
            "train/batch/p_loss": loss.item(),
            "train/batch/entropy": pred_action_dist.entropy().mean().item(),
        }


class ContinuousA2CAgent(ContinuousPGAgent):
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
