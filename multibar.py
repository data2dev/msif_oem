"""
Category C: Multi-bar Price-Volume Factors
==========================================
Rolling window technical factors across multiple lookback periods.

Features (24 total, 6 base features × 4 windows [5, 10, 20, 60]):
  For each window W in [5, 10, 20, 60]:
    - ret_W:      cumulative return over W bars
    - vol_W:      close-to-close volatility over W bars
    - rsi_W:      RSI over W bars
    - amihud_W:   Amihud illiquidity ratio over W bars
    - vret_W:     volume-return correlation over W bars
    - bbpos_W:    Bollinger Band %b position over W bars
"""

import numpy as np
import config as cfg


def compute(candles: np.ndarray) -> np.ndarray:
    """
    Compute multi-bar PV factors.
    
    Args:
        candles: [n_bars, 5] raw OHLCV (not scale-eliminated)
        
    Returns:
        [n_bars, 24] feature array
    """
    n = candles.shape[0]
    if n == 0:
        return np.zeros((0, N_FEATURES))

    closes = candles[:, 3]
    volumes = candles[:, 4]
    highs = candles[:, 1]
    lows = candles[:, 2]

    # Returns series
    returns = np.zeros(n)
    returns[1:] = (closes[1:] - closes[:-1]) / (closes[:-1] + 1e-10)

    windows = cfg.MULTIBAR_WINDOWS  # [5, 10, 20, 60]
    n_base = 6
    features = np.zeros((n, n_base * len(windows)))

    for wi, w in enumerate(windows):
        offset = wi * n_base

        for i in range(n):
            if i < w - 1:
                # Not enough history — leave as zero
                continue

            ret_slice = returns[i - w + 1:i + 1]
            close_slice = closes[i - w + 1:i + 1]
            vol_slice = volumes[i - w + 1:i + 1]

            # Cumulative return
            features[i, offset + 0] = np.sum(ret_slice)

            # Volatility (std of returns)
            features[i, offset + 1] = np.std(ret_slice) if len(ret_slice) > 1 else 0

            # RSI
            gains = np.maximum(ret_slice, 0)
            losses = np.maximum(-ret_slice, 0)
            avg_gain = np.mean(gains)
            avg_loss = np.mean(losses)
            if avg_loss > 0:
                rs = avg_gain / avg_loss
                features[i, offset + 2] = 1 - (1 / (1 + rs))
            else:
                features[i, offset + 2] = 1.0 if avg_gain > 0 else 0.5

            # Amihud illiquidity: mean(|return| / volume)
            vol_nonzero = vol_slice.copy()
            vol_nonzero[vol_nonzero == 0] = 1e-10
            amihud = np.mean(np.abs(ret_slice) / vol_nonzero)
            features[i, offset + 3] = np.log1p(amihud * 1e6)  # log-scaled

            # Volume-return correlation
            if np.std(ret_slice) > 0 and np.std(vol_slice) > 0:
                features[i, offset + 4] = np.corrcoef(ret_slice, vol_slice)[0, 1]
            else:
                features[i, offset + 4] = 0

            # Bollinger Band %b
            sma = np.mean(close_slice)
            std = np.std(close_slice)
            if std > 0:
                upper = sma + 2 * std
                lower = sma - 2 * std
                features[i, offset + 5] = (closes[i] - lower) / (upper - lower + 1e-10)
            else:
                features[i, offset + 5] = 0.5

    # Replace NaN/Inf
    features = np.nan_to_num(features, nan=0.0, posinf=3.0, neginf=-3.0)
    return features


N_FEATURES = 6 * len(cfg.MULTIBAR_WINDOWS)  # 24
