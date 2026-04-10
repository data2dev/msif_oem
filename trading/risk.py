"""
Risk Manager (RL-Integrated)
=============================
Safety guardrails that the RL agent cannot override:
  - Max position size cap per asset
  - Hard stop-loss per position
  - Daily drawdown circuit breaker
  - Sanity checks on RL actions

The RL agent decides allocations; the risk manager enforces limits.
"""

import copy
import logging
import time
import config as cfg

log = logging.getLogger(__name__)


class RiskManager:
    """Enforces risk limits on RL agent decisions."""

    def __init__(self):
        self.positions = {}
        self.daily_pnl = 0.0
        self.daily_start_equity = None
        self.last_reset_day = None
        self.halted = False

    def validate_rl_targets(self, targets: dict, equity: float,
                            prices: dict) -> dict:
        """
        Apply risk limits to RL agent's target positions.
        Clamps sizes, caps total exposure, zeroes everything if halted.
        """
        if self.halted:
            log.warning("Trading halted - zeroing all targets")
            return {sym: {"side": "flat", "target_size": 0.0, "target_frac": 0.0}
                    for sym in targets}

        validated = {}
        total_exposure = 0.0

        for sym, target in targets.items():
            price = prices.get(sym, 0)
            if price <= 0 or equity <= 0:
                validated[sym] = {"side": "flat", "target_size": 0.0, "target_frac": 0.0}
                continue

            side = target["side"]
            size = target["target_size"]
            max_size = (cfg.MAX_POSITION_FRAC * equity) / price
            size = min(size, max_size)
            exposure = size * price
            total_exposure += exposure
            validated[sym] = {
                "side": side,
                "target_size": size,
                "target_frac": target["target_frac"],
            }

        max_total = equity * 0.80
        if total_exposure > max_total and total_exposure > 0:
            scale = max_total / total_exposure
            for sym in validated:
                validated[sym]["target_size"] *= scale
                validated[sym]["target_frac"] *= scale

        return validated

    def check_stop_loss(self, symbol: str, current_price: float) -> bool:
        if symbol not in self.positions:
            return False
        pos = self.positions[symbol]
        entry = pos["entry_price"]
        if pos["side"] == "long":
            pnl_pct = (current_price - entry) / entry
        else:
            pnl_pct = (entry - current_price) / entry
        if pnl_pct < -cfg.STOP_LOSS_PCT:
            log.warning(f"STOP LOSS: {symbol} pnl={pnl_pct:.2%}")
            return True
        return False

    def can_close(self, symbol: str) -> bool:
        """Check if a position has been held long enough to close.
        Stop-loss bypasses this — use check_stop_loss separately."""
        if symbol not in self.positions:
            return True
        pos = self.positions[symbol]
        held_seconds = time.time() - pos.get("entry_time", 0)
        min_seconds = cfg.MIN_HOLD_MINUTES * 60
        if held_seconds < min_seconds:
            return False
        return True

    def update_pnl(self, realized_pnl: float, equity: float):
        today = time.strftime("%Y-%m-%d")
        if today != self.last_reset_day:
            self.daily_pnl = 0.0
            self.daily_start_equity = equity
            self.last_reset_day = today
            self.halted = False
        self.daily_pnl += realized_pnl
        if self.daily_start_equity and self.daily_start_equity > 0:
            drawdown = -self.daily_pnl / self.daily_start_equity
            if drawdown >= cfg.DAILY_DRAWDOWN_LIMIT:
                self.halted = True
                log.error(f"CIRCUIT BREAKER: drawdown {drawdown:.2%}")

    def register_position(self, symbol, side, size, entry_price):
        self.positions[symbol] = {
            "side": side, "size": size,
            "entry_price": entry_price, "entry_time": time.time(),
        }

    def close_position(self, symbol):
        self.positions.pop(symbol, None)

    def get_position(self, symbol):
        return self.positions.get(symbol)

    def get_positions_snapshot(self) -> dict:
        return copy.deepcopy(self.positions)
