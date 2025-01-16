from torch import nn


class DiscreteVNet(nn.Module):

    def __init__(self, observation_dim: int):
        super().__init__()
        self.layer1 = nn.Linear(observation_dim, 256)
        self.layer2 = nn.Linear(256, 128)
        self.layer3 = nn.Linear(128, 1)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.relu(self.layer1(x))
        x = self.relu(self.layer2(x))
        return self.layer3(x)
