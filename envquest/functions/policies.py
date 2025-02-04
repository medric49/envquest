from torch import nn


class DiscretePolicyNet(nn.Module):

    def __init__(self, observation_dim: int, num_actions: int):
        super().__init__()
        self.layer1 = nn.Linear(observation_dim, 256)
        self.layer2 = nn.Linear(256, 128)
        self.layer3 = nn.Linear(128, num_actions)
        self.relu = nn.ReLU()
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        x = self.relu(self.layer1(x))
        x = self.relu(self.layer2(x))
        x = self.softmax(self.layer3(x))
        return x


class ContinuousPolicyNet(nn.Module):

    def __init__(self, observation_dim: int, action_dim: int):
        super().__init__()
        self.layer1 = nn.Linear(observation_dim, 256)
        self.layer2 = nn.Linear(256, 128)
        self.layer3 = nn.Linear(128, action_dim)
        self.relu = nn.ReLU()
        self.tanh = nn.Tanh()

    def forward(self, x):
        x = self.relu(self.layer1(x))
        x = self.relu(self.layer2(x))
        x = self.tanh(self.layer3(x))
        return x
