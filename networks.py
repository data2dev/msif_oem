"""
SAC Networks
=============
Actor (Gaussian policy) and twin Q-networks for Soft Actor-Critic.
Designed for continuous action space: target portfolio weights per asset.

Action space: [-1, 1] per asset
  -1 = max short, 0 = flat, +1 = max long
  Scaled by MAX_POSITION_FRAC in the execution layer.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal
import numpy as np

LOG_STD_MIN = -20
LOG_STD_MAX = 2


class Actor(nn.Module):
    """
    Gaussian policy network.
    Outputs mean and log_std for each action dimension.
    Squashed through tanh to bound actions to [-1, 1].
    """

    def __init__(self, state_dim: int, action_dim: int, hidden: int = 256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
        )
        self.mean_head = nn.Linear(hidden, action_dim)
        self.log_std_head = nn.Linear(hidden, action_dim)

    def forward(self, state):
        """
        Returns:
            action: [B, action_dim] squashed actions
            log_prob: [B, 1] log probability
            mean: [B, action_dim] deterministic action (for eval)
        """
        h = self.net(state)
        mean = self.mean_head(h)
        log_std = self.log_std_head(h).clamp(LOG_STD_MIN, LOG_STD_MAX)
        std = log_std.exp()

        dist = Normal(mean, std)
        # Reparameterization trick
        x = dist.rsample()
        action = torch.tanh(x)

        # Log probability with tanh correction
        log_prob = dist.log_prob(x) - torch.log(1 - action.pow(2) + 1e-6)
        log_prob = log_prob.sum(dim=-1, keepdim=True)

        return action, log_prob, torch.tanh(mean)

    def get_action(self, state: np.ndarray, deterministic: bool = False) -> np.ndarray:
        """Get action from numpy state (for inference)."""
        with torch.no_grad():
            s = torch.FloatTensor(state).unsqueeze(0)
            action, _, mean = self.forward(s)
            if deterministic:
                return mean.squeeze(0).numpy()
            return action.squeeze(0).numpy()


class QNetwork(nn.Module):
    """Twin Q-network. Outputs two Q-values to mitigate overestimation."""

    def __init__(self, state_dim: int, action_dim: int, hidden: int = 256):
        super().__init__()
        # Q1
        self.q1 = nn.Sequential(
            nn.Linear(state_dim + action_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, 1),
        )
        # Q2
        self.q2 = nn.Sequential(
            nn.Linear(state_dim + action_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, 1),
        )

    def forward(self, state, action):
        """Returns (q1, q2) value estimates."""
        sa = torch.cat([state, action], dim=-1)
        return self.q1(sa), self.q2(sa)

    def q1_forward(self, state, action):
        """Single Q1 value (for policy gradient)."""
        sa = torch.cat([state, action], dim=-1)
        return self.q1(sa)
