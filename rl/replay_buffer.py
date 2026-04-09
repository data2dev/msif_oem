"""
Replay Buffer
==============
Stores (state, action, reward, next_state, done) transitions.

Supports outcome correction: when actual trade outcomes differ from
expected outcomes, the reward for that transition can be retroactively
updated with the real slippage/fee data.
"""

import numpy as np
import random
import logging

log = logging.getLogger(__name__)


class ReplayBuffer:
    """Experience replay with outcome correction."""

    def __init__(self, capacity: int = 100_000):
        self.capacity = capacity
        self.buffer = []
        self.position = 0

        # Outcome correction: maps transition index → pending correction
        self._pending_corrections = {}

    def push(self, state, action, reward, next_state, done) -> int:
        """
        Store a transition. Returns its index for later correction.
        """
        transition = {
            "state": np.array(state, dtype=np.float32),
            "action": np.array(action, dtype=np.float32),
            "reward": float(reward),
            "next_state": np.array(next_state, dtype=np.float32),
            "done": float(done),
            "corrected": False,
            "original_reward": float(reward),
        }

        if len(self.buffer) < self.capacity:
            self.buffer.append(transition)
            idx = len(self.buffer) - 1
        else:
            idx = self.position
            self.buffer[self.position] = transition
            self.position = (self.position + 1) % self.capacity

        return idx

    def correct_reward(self, idx: int, actual_reward: float):
        """
        Retroactively correct a transition's reward with actual outcome.
        Called by OutcomeTracker when real fill data arrives.
        """
        if 0 <= idx < len(self.buffer):
            old = self.buffer[idx]["reward"]
            self.buffer[idx]["reward"] = actual_reward
            self.buffer[idx]["corrected"] = True
            log.debug(f"Reward corrected: idx={idx}, {old:.6f} → {actual_reward:.6f}")

    def sample(self, batch_size: int) -> dict:
        """Sample a random batch."""
        batch_size = min(batch_size, len(self.buffer))
        batch = random.sample(self.buffer, batch_size)

        return {
            "states": np.stack([t["state"] for t in batch]),
            "actions": np.stack([t["action"] for t in batch]),
            "rewards": np.array([t["reward"] for t in batch], dtype=np.float32),
            "next_states": np.stack([t["next_state"] for t in batch]),
            "dones": np.array([t["done"] for t in batch], dtype=np.float32),
        }

    def __len__(self):
        return len(self.buffer)

    def save(self, path: str):
        """Save buffer to disk."""
        import pickle
        with open(path, "wb") as f:
            pickle.dump({
                "buffer": self.buffer,
                "position": self.position,
            }, f)
        log.info(f"Replay buffer saved: {len(self.buffer)} transitions → {path}")

    def load(self, path: str):
        """Load buffer from disk."""
        import pickle
        with open(path, "rb") as f:
            data = pickle.load(f)
        self.buffer = data["buffer"]
        self.position = data["position"]
        log.info(f"Replay buffer loaded: {len(self.buffer)} transitions ← {path}")
