"""
Category B: Intra-bar Price-Volume Factors
==========================================
Captures short-term intra-bar trading behavior.
Computed from each bar's OHLCV + trade data.

Features (10):
  0: twap          - (open + high + low + close) / 4, normalized
  1: vwap_ratio    - vwap / close
  2: bar_range     - (high - low) / close
  3: bar_return    - (close - open) / open
  4: upper_shadow  - (high - max(open, close)) / close
  5: lower_shadow  - (min(open, close) - low) / close
  6: body_ratio    - abs(close - open) / (high - low + 1e-10)
  7: volume_norm   - volume / rolling_mean_volume
  8: buy_ratio     - buy_vol / total_vol (from trades)
  9: trade_intensity - n_trades / mean_trades
"""

import numpy as np


def compute(bars: list) -> np.ndarray:
    """
    Compute intra-bar PV factors for a sequence of completed bars.
    
    Args:
        bars: list of bar_data dicts from DataStore
        
    Returns:
        [n_bars, 10] feature array
    """
    n = len(bars)
    if n == 0:
        return np.zeros((0, N_FEATURES))

    features = np.zeros((n, N_FEATURES))

    # Pre-compute rolling means for normalization
    volumes = np.array([b.get("volume", 0) for b in bars])
    n_trades_arr = np.array([b.get("n_trades", 0) for b in bars])

    vol_mean = _rolling_mean(volumes, 20)
    trade_mean = _rolling_mean(n_trades_arr, 20)

    for i, bar in enumerate(bars):
        o = bar.get("open", 0)
        h = bar.get("high", 0)
        l = bar.get("low", 0)
        c = bar.get("close", 0)
        v = bar.get("volume", 0)
        vwap = bar.get("vwap", c)
        buy_vol = bar.get("buy_vol", 0)
        sell_vol = bar.get("sell_vol", 0)
        n_trades = bar.get("n_trades", 0)

        if c <= 0:
            continue

        # TWAP normalized by close
        features[i, 0] = (o + h + l + c) / (4 * c) if c > 0 else 1.0

        # VWAP ratio
        features[i, 1] = vwap / c if c > 0 else 1.0

        # Bar range
        features[i, 2] = (h - l) / c if c > 0 else 0

        # Bar return
        features[i, 3] = (c - o) / o if o > 0 else 0

        # Upper shadow
        body_top = max(o, c)
        features[i, 4] = (h - body_top) / c if c > 0 else 0

        # Lower shadow
        body_bottom = min(o, c)
        features[i, 5] = (body_bottom - l) / c if c > 0 else 0

        # Body ratio
        bar_range = h - l
        features[i, 6] = abs(c - o) / (bar_range + 1e-10)

        # Volume normalized
        features[i, 7] = v / vol_mean[i] if vol_mean[i] > 0 else 1.0

        # Buy ratio (aggressor side from trades)
        total_trade_vol = buy_vol + sell_vol
        features[i, 8] = buy_vol / total_trade_vol if total_trade_vol > 0 else 0.5

        # Trade intensity
        features[i, 9] = n_trades / trade_mean[i] if trade_mean[i] > 0 else 1.0

    return features


def _rolling_mean(arr: np.ndarray, window: int) -> np.ndarray:
    """Compute rolling mean with expanding window for initial bars."""
    n = len(arr)
    result = np.ones(n)
    cumsum = 0
    for i in range(n):
        cumsum += arr[i]
        w = min(i + 1, window)
        start = max(0, i - window + 1)
        result[i] = np.mean(arr[start:i+1]) if w > 0 else 1.0
    return result


N_FEATURES = 10
