"""
RL State Builder
=================
Constructs the observation vector the RL agent sees each step.

State dimensions (per asset, N_ASSETS = len(PAIRS)):
  0:  alpha_signal       - ESGD predicted forward return
  1:  alpha_confidence   - ESGD cluster distance confidence
  2:  position_frac      - current position as fraction of equity (-1 to 1)
  3:  unrealized_pnl     - unrealized P&L as fraction of entry
  4:  time_held          - minutes since position opened (normalized)
  5:  spread_bps         - current bid-ask spread in bps
  6:  recent_slippage    - rolling mean slippage from last N fills
  7:  fill_rate          - fraction of recent limit orders that filled

Global dimensions (shared across assets):
  0:  equity_frac        - current equity / starting equity (drawdown gauge)
  1:  daily_pnl_frac     - today's realized P&L / starting equity
  2:  drawdown_remaining - how much drawdown budget is left (0=halted, 1=fresh)
  3:  regime_id          - ESGD cluster (one-hot encoded, K dims)
  4:  volatility_regime  - rolling 60-bar return volatility (normalized)
  5:  hour_sin           - sin(2π * hour/24) for time-of-day encoding
  6:  hour_cos           - cos(2π * hour/24)

Total state dim = N_ASSETS * 8 + 7 + K
"""

import time
import math
import numpy as np
import config as cfg


class StateBuilder:
    """Builds observation vectors for the RL agent."""

    def __init__(self, n_assets: int):
        self.n_assets = n_assets
        self.n_per_asset = 8
        self.n_global = 7 + cfg.N_CLUSTERS  # 7 base + K one-hot
        self.state_dim = n_assets * self.n_per_asset + self.n_global

        # Slippage / fill tracking (rolling windows)
        self.slippage_history = {i: [] for i in range(n_assets)}
        self.fill_history = {i: [] for i in range(n_assets)}
        self.max_history = 50

        # Starting equity for normalization
        self.starting_equity = None

    def build(self, signals: dict, symbols: list, risk_mgr,
              spreads: dict, equity: float, esgd_cluster: int,
              recent_volatility: float = 0.0) -> np.ndarray:
        """
        Build full state vector.

        Args:
            signals: {symbol: signal_dict} from SignalGenerator
            symbols: ordered list of symbols
            risk_mgr: RiskManager instance
            spreads: {symbol: spread_in_bps}
            equity: current portfolio equity
            esgd_cluster: current ESGD regime cluster id
            recent_volatility: 60-bar rolling return volatility

        Returns:
            state: [state_dim] numpy array
        """
        if self.starting_equity is None:
            self.starting_equity = equity

        state = np.zeros(self.state_dim, dtype=np.float32)

        # ── Per-asset features ──
        for i, sym in enumerate(symbols):
            offset = i * self.n_per_asset
            sig = signals.get(sym, {})

            # Alpha signal — clamp to [-1, 1] (raw can be enormous from ESGD)
            raw_alpha = sig.get("raw_alpha", 0.0)
            state[offset + 0] = np.clip(np.tanh(raw_alpha * 100), -1.0, 1.0)

            # Confidence — already 0-1 range
            state[offset + 1] = np.clip(sig.get("confidence", 0.0), 0.0, 1.0)

            # Position state
            pos = risk_mgr.get_position(sym)
            if pos and equity > 0:
                pos_value = pos["size"] * pos.get("entry_price", 0)
                sign = 1.0 if pos["side"] == "long" else -1.0
                state[offset + 2] = np.clip(sign * pos_value / equity, -1.0, 1.0)

                # Unrealized P&L — clamp
                entry = pos["entry_price"]
                state[offset + 3] = np.clip(sig.get("raw_alpha", 0.0) * sign, -1.0, 1.0)

                # Time held (normalized: 0=just opened, 1=held for 60 mins)
                elapsed = (time.time() - pos.get("entry_time", time.time())) / 3600
                state[offset + 4] = min(elapsed, 1.0)
            else:
                state[offset + 2] = 0.0
                state[offset + 3] = 0.0
                state[offset + 4] = 0.0

            # Spread
            state[offset + 5] = spreads.get(sym, 0.0) / 100.0  # normalize bps

            # Recent slippage (rolling mean)
            slip_hist = self.slippage_history.get(i, [])
            state[offset + 6] = np.mean(slip_hist) if slip_hist else 0.0

            # Fill rate
            fill_hist = self.fill_history.get(i, [])
            state[offset + 7] = np.mean(fill_hist) if fill_hist else 0.5

        # ── Global features ──
        g_offset = self.n_assets * self.n_per_asset

        # Equity fraction (drawdown gauge)
        if self.starting_equity > 0:
            state[g_offset + 0] = equity / self.starting_equity
        else:
            state[g_offset + 0] = 1.0

        # Daily P&L fraction
        if self.starting_equity > 0:
            state[g_offset + 1] = risk_mgr.daily_pnl / self.starting_equity
        else:
            state[g_offset + 1] = 0.0

        # Drawdown budget remaining
        if risk_mgr.halted:
            state[g_offset + 2] = 0.0
        elif self.starting_equity > 0:
            used = max(-risk_mgr.daily_pnl / self.starting_equity, 0)
            state[g_offset + 2] = max(1.0 - used / cfg.DAILY_DRAWDOWN_LIMIT, 0)
        else:
            state[g_offset + 2] = 1.0

        # Regime one-hot
        regime_start = g_offset + 3
        if 0 <= esgd_cluster < cfg.N_CLUSTERS:
            state[regime_start + esgd_cluster] = 1.0

        # Volatility regime
        state[g_offset + 3 + cfg.N_CLUSTERS] = min(recent_volatility * 100, 3.0)

        # Time-of-day encoding
        hour = time.gmtime().tm_hour + time.gmtime().tm_min / 60.0
        state[g_offset + 4 + cfg.N_CLUSTERS] = math.sin(2 * math.pi * hour / 24)
        state[g_offset + 5 + cfg.N_CLUSTERS] = math.cos(2 * math.pi * hour / 24)

        # Final safety: clamp entire state to [-5, 5] to prevent RL divergence
        state = np.clip(state, -5.0, 5.0)
        state = np.nan_to_num(state, nan=0.0, posinf=5.0, neginf=-5.0)

        return state

    def record_fill(self, asset_idx: int, expected_price: float,
                    actual_price: float, filled: bool):
        """Record trade outcome for slippage/fill rate tracking."""
        # Slippage
        if expected_price > 0 and actual_price > 0:
            slippage = abs(actual_price - expected_price) / expected_price
            hist = self.slippage_history[asset_idx]
            hist.append(slippage)
            if len(hist) > self.max_history:
                hist.pop(0)

        # Fill rate
        hist = self.fill_history[asset_idx]
        hist.append(1.0 if filled else 0.0)
        if len(hist) > self.max_history:
            hist.pop(0)

    def update_position_pnl(self, asset_idx: int, state: np.ndarray,
                            actual_pnl_frac: float) -> np.ndarray:
        """Correct the unrealized P&L in state with actual value."""
        offset = asset_idx * self.n_per_asset + 3
        state[offset] = actual_pnl_frac
        return state
