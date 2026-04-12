"""
Reward Function
================
Shaped reward for the RL execution agent.

Two-stage reward:
  1. EXPECTED reward: computed immediately from predicted fill prices
  2. ACTUAL reward: computed later when real fill data arrives

The difference between expected and actual teaches the agent about
slippage, partial fills, and fee structure changes over time.

Reward components:
  r = Δportfolio_value
      - λ_fee  * |transaction_costs|
      - λ_slip * |slippage_cost|
      - λ_dd   * max(0, drawdown_penalty)
      + λ_hold * holding_cost_penalty    (penalize long idle holds)
      + λ_flat * flat_bonus              (small reward for staying flat when alpha is weak)
"""

import numpy as np
import numpy as np
import config as cfg


# Reward shaping weights
LAMBDA_FEE = 2.0       # transaction cost penalty multiplier
LAMBDA_SLIP = 3.0      # slippage penalty (heavier — agent should learn to avoid it)
LAMBDA_DD = 5.0        # drawdown penalty (heaviest — protect capital)
LAMBDA_HOLD = 0.0001   # tiny penalty per bar for holding (encourages decisiveness)
LAMBDA_FLAT = 0.0002   # tiny reward for staying flat when signal is weak


class RewardCalculator:
    """Computes shaped rewards for the RL agent."""

    def __init__(self):
        self.taker_fee_rate = 0.0026   # Kraken taker fee ~0.26%
        self.maker_fee_rate = 0.0016   # Kraken maker fee ~0.16%

    def compute_expected(self, prev_positions: dict, new_positions: dict,
                         prices: dict, equity: float,
                         daily_pnl: float, starting_equity: float) -> float:
        """
        Compute expected reward based on intended trades.
        Called immediately when the agent acts.
        
        Args:
            prev_positions: {symbol: {"side", "size", "entry_price"}} before action
            new_positions: {symbol: {"side", "size", "entry_price"}} after action
            prices: {symbol: current_price}
            equity: current equity
            daily_pnl: today's cumulative P&L
            starting_equity: equity at start of day
            
        Returns:
            expected_reward: float
        """
        reward = 0.0

        for sym in set(list(prev_positions.keys()) + list(new_positions.keys())):
            price = prices.get(sym, 0)
            if price <= 0:
                continue

            prev = prev_positions.get(sym)
            curr = new_positions.get(sym)

            # ── P&L change ──
            prev_value = 0.0
            if prev:
                sign = 1.0 if prev["side"] == "long" else -1.0
                prev_value = sign * prev["size"] * (price - prev["entry_price"])

            curr_value = 0.0
            if curr:
                sign = 1.0 if curr["side"] == "long" else -1.0
                curr_value = sign * curr["size"] * (price - curr["entry_price"])

            pnl_delta = curr_value - prev_value

            # ── Transaction cost ──
            trade_volume = 0.0
            if prev and curr:
                # Changed position
                trade_volume = abs(curr["size"] - prev["size"]) * price
            elif curr and not prev:
                # Opened new position
                trade_volume = curr["size"] * price
            elif prev and not curr:
                # Closed position
                trade_volume = prev["size"] * price

            fee_cost = trade_volume * self.maker_fee_rate  # assume limit orders

            # ── Holding cost (anti-stagnation) ──
            hold_penalty = 0.0
            if curr:
                hold_penalty = LAMBDA_HOLD * curr["size"] * price

            # ── Flat bonus ──
            flat_bonus = 0.0
            if not curr and not prev:
                flat_bonus = LAMBDA_FLAT

            reward += pnl_delta - LAMBDA_FEE * fee_cost - hold_penalty + flat_bonus

        # ── Drawdown penalty ──
        if starting_equity > 0:
            current_dd = max(-daily_pnl / starting_equity, 0)
            dd_threshold = cfg.DAILY_DRAWDOWN_LIMIT * 0.5  # start penalizing at 50%
            if current_dd > dd_threshold:
                excess = current_dd - dd_threshold
                reward -= LAMBDA_DD * excess * equity

        # Normalize by equity to make reward scale-invariant
        if equity > 0:
            reward /= equity

        # Clamp to prevent RL divergence from extreme values
        return float(np.clip(reward, -1.0, 1.0))

    def compute_actual(self, expected_reward: float, expected_fills: dict,
                       actual_fills: dict, prices: dict, equity: float) -> float:
        """
        Compute actual reward by correcting expected with real fill data.
        Called when fill confirmations arrive from exchange.
        
        Args:
            expected_reward: the reward computed at action time
            expected_fills: {symbol: {"price", "volume", "side"}} intended
            actual_fills: {symbol: {"price", "volume", "side", "filled"}} real
            prices: current prices at correction time
            equity: equity at correction time
            
        Returns:
            actual_reward: corrected reward
        """
        correction = 0.0

        for sym in expected_fills:
            exp = expected_fills[sym]
            act = actual_fills.get(sym, {})

            exp_price = exp.get("price", 0)
            act_price = act.get("price", exp_price)
            exp_vol = exp.get("volume", 0)
            act_vol = act.get("volume", 0)
            filled = act.get("filled", False)

            if not filled:
                # Order didn't fill — revert the expected P&L
                # The agent needs to learn that limit orders sometimes miss
                sign = 1.0 if exp.get("side") == "buy" else -1.0
                missed_exposure = exp_vol * exp_price
                correction -= abs(expected_reward) * 0.5  # partial penalty for missed fill

            elif exp_price > 0 and act_price > 0:
                # Slippage correction
                slippage = abs(act_price - exp_price)
                slippage_cost = slippage * act_vol
                correction -= LAMBDA_SLIP * slippage_cost

                # Volume difference (partial fill)
                if exp_vol > 0 and act_vol < exp_vol:
                    fill_ratio = act_vol / exp_vol
                    # Scale reward proportionally to actual fill
                    correction += expected_reward * (fill_ratio - 1.0) * 0.5

                # Fee adjustment (taker vs maker)
                if act.get("was_taker", False):
                    extra_fee = (self.taker_fee_rate - self.maker_fee_rate) * act_vol * act_price
                    correction -= LAMBDA_FEE * extra_fee

        # Normalize
        if equity > 0:
            correction /= equity

        return float(np.clip(expected_reward + correction, -1.0, 1.0))

    def update_fee_rates(self, taker: float, maker: float):
        """Update fee rates when they change (volume tier changes on Kraken)."""
        self.taker_fee_rate = taker
        self.maker_fee_rate = maker
