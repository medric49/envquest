import torch.nn as nn
from torch import distributions
import torch
from torch.distributions.utils import _standard_normal


class DiscreteQNet(nn.Module):

    def __init__(self, observation_dim: int, num_actions: int):
        super(DiscreteQNet, self).__init__()
        self.layer1 = nn.Linear(observation_dim, 128)
        self.layer2 = nn.Linear(128, 128)
        self.layer3 = nn.Linear(128, num_actions)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.relu(self.layer1(x))
        x = self.relu(self.layer2(x))
        return self.layer3(x)


class TruncatedNormal(distributions.Normal):
    def __init__(self, loc, scale, low=0.0, high=1.0, eps=1e-6):
        super().__init__(loc, scale, validate_args=False)
        self.low = low
        self.high = high
        self.eps = eps

    def _clip(self, x):
        clipped_x = torch.clip(x, self.low + self.eps, self.high - self.eps)
        x = x - x.detach() + clipped_x.detach()
        return x

    def sample(self, clip=None, sample_shape=torch.Size()):
        shape = self._extended_shape(sample_shape)
        eps = _standard_normal(shape, dtype=self.loc.dtype, device=self.loc.device)
        eps *= self.scale
        if clip is not None:
            eps = torch.clip(eps, -clip, clip)
        x = self.loc + eps
        return self._clip(x)


class DiscretePolicyNet(nn.Module):

    def __init__(self, observation_dim: int, num_actions: int):
        super().__init__()

        self.layer1 = nn.Linear(observation_dim, 128)
        self.bn1 = nn.BatchNorm1d(128)
        self.layer2 = nn.Linear(128, 128)
        self.bn2 = nn.BatchNorm1d(128)
        self.layer3 = nn.Linear(128, num_actions)
        self.relu = nn.ReLU()
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        x = self.relu(self.bn1(self.layer1(x)))
        x = self.relu(self.bn2(self.layer2(x)))
        action_mean = self.softmax(self.layer3(x))
        action_dist = distributions.Categorical(action_mean)
        return action_dist


class ValueNet(nn.Module):

    def __init__(self, observation_dim: int):
        super().__init__()

        self.layer1 = nn.Linear(observation_dim, 128)
        self.bn1 = nn.BatchNorm1d(128)
        self.layer2 = nn.Linear(128, 128)
        self.bn2 = nn.BatchNorm1d(128)
        self.layer3 = nn.Linear(128, 1)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.relu(self.bn1(self.layer1(x)))
        x = self.relu(self.bn2(self.layer2(x)))
        x = self.layer3(x)
        return x
