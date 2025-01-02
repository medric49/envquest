import random
import numpy as np
import torch


def device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def set_seed_everywhere(seed):
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)


def standardize(x, mean, std, eps=1e-8):
    return (x - mean) / (std + eps)


def normalize(x, x_min, x_max, eps=1e-6):
    return (x - x_min) / (x_max - x_min + eps)


class Until:
    def __init__(self, until):
        self._until = until

    def __call__(self, step):
        if self._until is None:
            return True
        until = self._until
        return step < until


class Every:
    def __init__(self, every):
        self._every = every

    def __call__(self, step):
        if self._every is None:
            return False
        every = self._every
        if step % every == 0:
            return True
        return False


def init_weights(m):
    if isinstance(m, torch.nn.Linear):
        torch.nn.init.xavier_uniform(m.weight)
        m.bias.data.fill_(0.0)
