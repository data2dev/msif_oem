"""
SAC Agent
==========
Soft Actor-Critic for continuous portfolio allocation.

Key features:
  - Automatic entropy temperature (α) tuning
  - Twin Q-networks to reduce overestimation
  - Soft target updates (Polyak averaging)
  - Shadow mode: outputs suggestions alongside hardcoded rules
  - Warmup period: random actions until replay buffer has enough data

The agent outputs target allocation weights per asset: [-1, 1]
  -1 = max short allocation (MAX_POSITION_FRAC of equity)
   0 = flat (no position)
  +1 = max long allocation
"""

import os
import copy
import logging
import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim
from rl.networks import Actor, QNetwork
from rl.replay_buffer import ReplayBuffer

log = logging.getLogger(__name__)


class SACAgent:
    """Soft Actor-Critic agent for portfolio execution."""

    def __init__(self, state_dim: int, action_dim: int,
                 hidden_dim: int = 256, lr: float = 3e-4,
                 gamma: float = 0.99, tau: float = 0.005,
                 buffer_capacity: int = 100_000,
                 batch_size: int = 256, warmup_steps: int = 1000,
                 device: str = None):
        """
        Args:
            state_dim: observation dimension
            action_dim: number of assets (action per asset)
            hidden_dim: hidden layer size for networks
            lr: learning rate
            gamma: discount factor
            tau: soft target update rate
            buffer_capacity: replay buffer size
            batch_size: training batch size
            warmup_steps: random actions before training starts
            device: torch device
        """
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.gamma = gamma
        self.tau = tau
        self.batch_size = batch_size
        self.warmup_steps = warmup_steps

        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)

        # ── Networks ──
        self.actor = Actor(state_dim, action_dim, hidden_dim).to(self.device)
        self.critic = QNetwork(state_dim, action_dim, hidden_dim).to(self.device)
        self.critic_target = copy.deepcopy(self.critic).to(self.device)

        # Freeze target (only updated via Polyak)
        for p in self.critic_target.parameters():
            p.requires_grad = False

        # ── Optimizers ──
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=lr)
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=lr)

        # ── Automatic entropy tuning ──
        self.target_entropy = -action_dim  # heuristic: -dim(A)
        self.log_alpha = torch.zeros(1, requires_grad=True, device=self.device)
        self.alpha_optimizer = optim.Adam([self.log_alpha], lr=lr)

        # ── Replay buffer ──
        self.replay_buffer = ReplayBuffer(capacity=buffer_capacity)

        # ── Step counter ──
        self.total_steps = 0
        self.training_steps = 0

        log.info(f"SAC Agent: state_dim={state_dim}, action_dim={action_dim}, "
                 f"device={self.device}")
        log.info(f"  Networks: {self._count_params()} total parameters")
        log.info(f"  Warmup: {warmup_steps} steps of random exploration")

    @property
    def alpha(self):
        return self.log_alpha.exp().item()

    def select_action(self, state: np.ndarray, deterministic: bool = False) -> np.ndarray:
        """
        Select action given current state. LONG-ONLY: actions clamped to [0, 1].
        
        During warmup: random actions in [0, 1].
        After warmup: policy network output, clamped to [0, 1].
        
        Args:
            state: [state_dim] numpy array
            deterministic: if True, use mean action (no exploration)
            
        Returns:
            action: [action_dim] numpy array in [0, 1]
        """
        if self.total_steps < self.warmup_steps and not deterministic:
            return np.random.uniform(0, 1, self.action_dim).astype(np.float32)

        action = self.actor.get_action(state, deterministic=deterministic)
        return np.clip(action, 0, 1)

    def step(self, state, action, reward, next_state, done) -> int:
        """
        Store transition and train if ready.
        
        Returns:
            transition index in replay buffer
        """
        # Clamp reward to prevent divergence
        reward = float(np.clip(reward, -1.0, 1.0))
        idx = self.replay_buffer.push(state, action, reward, next_state, done)
        self.total_steps += 1

        # Train after warmup
        if (self.total_steps >= self.warmup_steps and
                len(self.replay_buffer) >= self.batch_size):
            self._train_step()

        return idx

    def _train_step(self):
        """One SAC training step."""
        try:
            batch = self.replay_buffer.sample(self.batch_size)

            states = torch.FloatTensor(batch["states"]).to(self.device)
            actions = torch.FloatTensor(batch["actions"]).to(self.device)
            rewards = torch.FloatTensor(batch["rewards"]).unsqueeze(1).to(self.device)
            next_states = torch.FloatTensor(batch["next_states"]).to(self.device)
            dones = torch.FloatTensor(batch["dones"]).unsqueeze(1).to(self.device)

            # Check for NaN in batch
            if (torch.isnan(states).any() or torch.isnan(actions).any() or
                    torch.isnan(rewards).any() or torch.isnan(next_states).any()):
                log.warning(f"NaN detected in batch — skipping training step. "
                            f"states_nan={torch.isnan(states).sum()}, "
                            f"actions_nan={torch.isnan(actions).sum()}, "
                            f"rewards_nan={torch.isnan(rewards).sum()}")
                self.training_steps += 1  # count it to avoid getting stuck
                return

            alpha = self.log_alpha.exp().detach()

            # ── Update Critics ──
            with torch.no_grad():
                next_actions, next_log_probs, _ = self.actor(next_states)
                q1_target, q2_target = self.critic_target(next_states, next_actions)
                q_target = torch.min(q1_target, q2_target) - alpha * next_log_probs
                td_target = rewards + (1 - dones) * self.gamma * q_target

            q1, q2 = self.critic(states, actions)
            critic_loss = F.mse_loss(q1, td_target) + F.mse_loss(q2, td_target)

            self.critic_optimizer.zero_grad()
            critic_loss.backward()
            torch.nn.utils.clip_grad_norm_(self.critic.parameters(), 1.0)
            self.critic_optimizer.step()

            # ── Update Actor ──
            new_actions, log_probs, _ = self.actor(states)
            q1_new = self.critic.q1_forward(states, new_actions)
            actor_loss = (alpha * log_probs - q1_new).mean()

            self.actor_optimizer.zero_grad()
            actor_loss.backward()
            torch.nn.utils.clip_grad_norm_(self.actor.parameters(), 1.0)
            self.actor_optimizer.step()

            # ── Update Alpha (entropy temperature) ──
            alpha_loss = -(self.log_alpha * (log_probs + self.target_entropy).detach()).mean()

            self.alpha_optimizer.zero_grad()
            alpha_loss.backward()
            self.alpha_optimizer.step()

            # ── Soft update target network ──
            for target_param, param in zip(self.critic_target.parameters(),
                                           self.critic.parameters()):
                target_param.data.copy_(
                    self.tau * param.data + (1 - self.tau) * target_param.data
                )

            self.training_steps += 1

            if self.training_steps % 100 == 0:
                log.info(
                    f"SAC step {self.training_steps}: "
                    f"critic_loss={critic_loss.item():.4f}, "
                    f"actor_loss={actor_loss.item():.4f}, "
                    f"alpha={self.alpha:.4f}, "
                    f"buffer={len(self.replay_buffer)}"
                )
            elif self.training_steps <= 3:
                log.info(
                    f"SAC training started! step={self.training_steps}, "
                    f"critic_loss={critic_loss.item():.4f}, "
                    f"actor_loss={actor_loss.item():.4f}"
                )

        except Exception as e:
            log.error(f"SAC _train_step CRASHED: {e}", exc_info=True)
            self.training_steps += 1  # don't get stuck retrying the same crash

    def action_to_positions(self, action: np.ndarray, symbols: list,
                            equity: float, prices: dict) -> dict:
        """
        Convert agent action to target position sizes. LONG-ONLY.
        
        Args:
            action: [n_assets] array in [0, 1]
            symbols: ordered list of asset symbols
            equity: current portfolio equity
            prices: {symbol: current_price}
            
        Returns:
            {symbol: {"side": str, "target_size": float, "target_frac": float}}
        """
        import config as cfg
        targets = {}

        for i, sym in enumerate(symbols):
            weight = float(action[i])
            price = prices.get(sym, 0)
            if price <= 0:
                continue

            # Scale by max position fraction
            frac = weight * cfg.MAX_POSITION_FRAC
            dollar_size = equity * frac
            volume = dollar_size / price

            if weight < 0.15:  # wide dead zone — only trade on strong conviction
                targets[sym] = {"side": "flat", "target_size": 0.0, "target_frac": 0.0}
            else:
                targets[sym] = {
                    "side": "long",
                    "target_size": volume,
                    "target_frac": frac,
                }

        return targets

    def _count_params(self) -> int:
        total = sum(p.numel() for p in self.actor.parameters())
        total += sum(p.numel() for p in self.critic.parameters())
        return total

    # ── Persistence ────────────────────────────

    def save(self, directory: str):
        os.makedirs(directory, exist_ok=True)
        torch.save({
            "actor": self.actor.state_dict(),
            "critic": self.critic.state_dict(),
            "critic_target": self.critic_target.state_dict(),
            "actor_opt": self.actor_optimizer.state_dict(),
            "critic_opt": self.critic_optimizer.state_dict(),
            "log_alpha": self.log_alpha.data,
            "alpha_opt": self.alpha_optimizer.state_dict(),
            "total_steps": self.total_steps,
            "training_steps": self.training_steps,
        }, os.path.join(directory, "sac_agent.pt"))

        self.replay_buffer.save(os.path.join(directory, "replay_buffer.pkl"))
        log.info(f"SAC agent saved to {directory}")

    def load(self, directory: str):
        path = os.path.join(directory, "sac_agent.pt")
        if not os.path.exists(path):
            log.warning(f"No saved agent at {path} — starting fresh")
            return

        checkpoint = torch.load(path, map_location=self.device, weights_only=False)
        self.actor.load_state_dict(checkpoint["actor"])
        self.critic.load_state_dict(checkpoint["critic"])
        self.critic_target.load_state_dict(checkpoint["critic_target"])
        self.actor_optimizer.load_state_dict(checkpoint["actor_opt"])
        self.critic_optimizer.load_state_dict(checkpoint["critic_opt"])
        self.log_alpha.data = checkpoint["log_alpha"]
        self.alpha_optimizer.load_state_dict(checkpoint["alpha_opt"])
        self.total_steps = checkpoint["total_steps"]
        self.training_steps = checkpoint["training_steps"]

        buf_path = os.path.join(directory, "replay_buffer.pkl")
        if os.path.exists(buf_path):
            self.replay_buffer.load(buf_path)

        log.info(f"SAC agent loaded: {self.total_steps} total steps, "
                 f"{self.training_steps} training steps, "
                 f"buffer={len(self.replay_buffer)}")
