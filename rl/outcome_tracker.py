"""
Outcome Tracker
================
Monitors what actually happens after the RL agent acts.

Flow:
  1. Agent decides action → executor places orders
  2. OutcomeTracker records the EXPECTED outcome (intended fills)
  3. Later, when fill data arrives from exchange, tracker computes ACTUAL outcome
  4. The reward difference is fed back into the replay buffer
  5. State builder gets updated slippage/fill stats

This is the key feedback loop that teaches the agent about real-world
execution quality — something no simulator can perfectly model.
"""

import time
import logging
import numpy as np
from rl.reward import RewardCalculator

log = logging.getLogger(__name__)


class PendingOutcome:
    """A trade action waiting for fill confirmation."""

    def __init__(self, transition_idx: int, expected_fills: dict,
                 expected_reward: float, action: np.ndarray,
                 state: np.ndarray, timestamp: float,
                 prev_positions: dict, equity: float):
        self.transition_idx = transition_idx
        self.expected_fills = expected_fills
        self.expected_reward = expected_reward
        self.action = action
        self.state = state
        self.timestamp = timestamp
        self.prev_positions = prev_positions
        self.equity = equity
        self.resolved = False


class OutcomeTracker:
    """
    Tracks pending trade outcomes and feeds corrections back
    into the replay buffer when actual fill data arrives.
    """

    def __init__(self, replay_buffer, state_builder, reward_calc: RewardCalculator):
        self.replay_buffer = replay_buffer
        self.state_builder = state_builder
        self.reward_calc = reward_calc
        self.pending = []  # list of PendingOutcome

        # Statistics
        self.total_corrections = 0
        self.total_slippage = 0.0
        self.total_missed_fills = 0
        self.total_fills = 0

        # Timeout: if no fill data after this many seconds, assume missed
        self.fill_timeout = 120  # 2 minutes

    def register_action(self, transition_idx: int, expected_fills: dict,
                        expected_reward: float, action: np.ndarray,
                        state: np.ndarray, prev_positions: dict,
                        equity: float):
        """
        Register a new action for outcome tracking.
        
        Args:
            transition_idx: index in replay buffer
            expected_fills: {symbol: {"price", "volume", "side"}}
            expected_reward: reward computed at action time
            action: agent's action array
            state: state at action time
            prev_positions: positions before action
            equity: equity at action time
        """
        if not expected_fills:
            return

        pending = PendingOutcome(
            transition_idx=transition_idx,
            expected_fills=expected_fills,
            expected_reward=expected_reward,
            action=action,
            state=state,
            timestamp=time.time(),
            prev_positions=prev_positions,
            equity=equity,
        )
        self.pending.append(pending)
        log.debug(f"Registered pending outcome: {len(expected_fills)} fills, "
                  f"idx={transition_idx}")

    def resolve_fills(self, actual_fills: dict, prices: dict):
        """
        Called when fill data arrives from the exchange.
        
        Args:
            actual_fills: {symbol: {"price", "volume", "side", "filled", "was_taker", "txid"}}
            prices: {symbol: current_price}
        """
        resolved_indices = []

        for i, pending in enumerate(self.pending):
            if pending.resolved:
                continue

            # Check if any of this pending's expected fills match the actual fills
            matched = False
            for sym in pending.expected_fills:
                if sym in actual_fills:
                    matched = True
                    break

            if not matched:
                continue

            # Compute actual reward
            actual_reward = self.reward_calc.compute_actual(
                expected_reward=pending.expected_reward,
                expected_fills=pending.expected_fills,
                actual_fills=actual_fills,
                prices=prices,
                equity=pending.equity,
            )

            # Correct the replay buffer
            self.replay_buffer.correct_reward(pending.transition_idx, actual_reward)

            # Update state builder with slippage/fill data
            for sym_idx, sym in enumerate(pending.expected_fills.keys()):
                exp = pending.expected_fills[sym]
                act = actual_fills.get(sym, {})

                exp_price = exp.get("price", 0)
                act_price = act.get("price", exp_price)
                filled = act.get("filled", False)

                self.state_builder.record_fill(
                    asset_idx=sym_idx,
                    expected_price=exp_price,
                    actual_price=act_price,
                    filled=filled,
                )

                if filled:
                    self.total_fills += 1
                    if exp_price > 0:
                        self.total_slippage += abs(act_price - exp_price) / exp_price
                else:
                    self.total_missed_fills += 1

            # Track stats
            reward_diff = actual_reward - pending.expected_reward
            self.total_corrections += 1

            if abs(reward_diff) > 0.0001:
                log.info(
                    f"Outcome correction #{self.total_corrections}: "
                    f"expected={pending.expected_reward:.6f}, "
                    f"actual={actual_reward:.6f}, "
                    f"diff={reward_diff:+.6f}"
                )

            pending.resolved = True
            resolved_indices.append(i)

        # Clean resolved
        self.pending = [p for p in self.pending if not p.resolved]

    def check_timeouts(self, prices: dict):
        """
        Handle pending outcomes that have timed out (no fill confirmation).
        Assumes the order didn't fill — penalizes the agent.
        """
        now = time.time()
        timed_out = []

        for pending in self.pending:
            if pending.resolved:
                continue
            if now - pending.timestamp > self.fill_timeout:
                timed_out.append(pending)

        for pending in timed_out:
            # Build "missed fill" actual data
            actual_fills = {}
            for sym, exp in pending.expected_fills.items():
                actual_fills[sym] = {
                    "price": exp.get("price", 0),
                    "volume": 0,
                    "side": exp.get("side", ""),
                    "filled": False,
                    "was_taker": False,
                }

            actual_reward = self.reward_calc.compute_actual(
                expected_reward=pending.expected_reward,
                expected_fills=pending.expected_fills,
                actual_fills=actual_fills,
                prices=prices,
                equity=pending.equity,
            )

            self.replay_buffer.correct_reward(pending.transition_idx, actual_reward)

            # Update fill rate stats
            for sym_idx, sym in enumerate(pending.expected_fills.keys()):
                self.state_builder.record_fill(
                    asset_idx=sym_idx,
                    expected_price=pending.expected_fills[sym].get("price", 0),
                    actual_price=0,
                    filled=False,
                )
                self.total_missed_fills += 1

            pending.resolved = True
            self.total_corrections += 1
            log.warning(f"Fill timeout for transition {pending.transition_idx}")

        self.pending = [p for p in self.pending if not p.resolved]

    def get_stats(self) -> dict:
        """Get outcome tracking statistics."""
        avg_slippage = (self.total_slippage / self.total_fills
                        if self.total_fills > 0 else 0)
        fill_rate = (self.total_fills / (self.total_fills + self.total_missed_fills)
                     if (self.total_fills + self.total_missed_fills) > 0 else 0)
        return {
            "total_corrections": self.total_corrections,
            "total_fills": self.total_fills,
            "total_missed": self.total_missed_fills,
            "avg_slippage_pct": avg_slippage * 100,
            "fill_rate": fill_rate,
            "pending": len(self.pending),
        }
