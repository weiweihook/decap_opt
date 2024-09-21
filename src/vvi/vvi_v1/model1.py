import numpy as np
import torch
import torch.nn as nn

from torch.distributions.categorical import Categorical
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

# cuda = True
# device = torch.device("cuda" if torch.cuda.is_available() and cuda else "cpu")
device = 'cpu'
EPS = 1e-5

def layer_init(layer, std=np.sqrt(2), bias_const=0.0):
    torch.nn.init.orthogonal_(layer.weight, std)
    torch.nn.init.constant_(layer.bias, bias_const)
    return layer

class Transpose(nn.Module):
    def __init__(self, permutation):
        super().__init__()
        self.permutation = permutation

    def forward(self, x):
        return x.permute(self.permutation)


class CategoricalMasked(Categorical):
    def __init__(self, probs=None, logits=None, validate_args=None, masks=[]):
        self.masks = masks
        if len(self.masks) == 0:
            super(CategoricalMasked, self).__init__(probs, logits, validate_args)
        else:
            self.masks = masks.type(torch.BoolTensor).to(device)
            logits = torch.where(self.masks, logits, torch.tensor(-1e+8).to(device))
            super(CategoricalMasked, self).__init__(probs, logits, validate_args)

    def entropy(self):
        if len(self.masks) == 0:
            return super(CategoricalMasked, self).entropy()
        p_log_p = self.logits * self.probs
        p_log_p = torch.where(self.masks, p_log_p, torch.tensor(0.0).to(device))
        return -p_log_p.sum(-1)


class PPONetwork(nn.Module):
    def __init__(self, env):
        super(PPONetwork, self).__init__()
        self.network = nn.Sequential(
            Transpose((0, 1, 2, 3)),
            layer_init(nn.Conv2d(4, 16, kernel_size=3, stride=2)),
            nn.ReLU(),
            layer_init(nn.Conv2d(16, 32, kernel_size=2)),
            nn.ReLU(),
            nn.Flatten(),
            layer_init(nn.Linear(128, 64)),
            nn.ReLU(),
        )
        self.env = env
        self.action_space = env.action_space
        self.actor = layer_init(nn.Linear(64, self.action_space.sum()), std=0.01)
        self.critic = layer_init(nn.Linear(64, 1), std=1)

    def get_value(self, x):
        x0 = self.network(x)

        return self.critic(x0)

    def get_action_and_value(self, x, action_mask, action=None):
        x0 = self.network(x)
        hidden = x0
        logits = self.actor(hidden)
        # print(x0, logits)
        split_logits = torch.split(logits, self.action_space.tolist(), dim=1)
        split_action_masks = torch.split(action_mask, self.action_space.tolist(), dim=1)
        multi_categoricals = [
            CategoricalMasked(logits=logits, masks=iam) for (logits, iam) in zip(split_logits, split_action_masks)
        ]
        if action is None:
            action = torch.stack([categorical.sample() for categorical in multi_categoricals])
        logprob = torch.stack([categorical.log_prob(a) for a, categorical in zip(action, multi_categoricals)])
        entropy = torch.stack([categorical.entropy() for categorical in multi_categoricals])
        return action.T, logprob.sum(0), entropy.sum(0), self.critic(hidden)

