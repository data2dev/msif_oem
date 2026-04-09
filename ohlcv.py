"""
Category A: Raw OHLCV Features
===============================
5 features: open, high, low, close, volume
Scale-eliminated per the paper: prices / last_close, volume / last_volume.
"""

import numpy as np


def compute(candles: np.ndarray) -> np.ndarray:
    """
    Compute scale-eliminated OHLCV features.
    
    Args:
        candles: [n_bars, 5] array of (open, high, low, close, volume)
        
    Returns:
        [n_bars, 5] normalized OHLCV
    """
    if candles.shape[0] == 0:
        return np.zeros((0, 5))

    result = candles.copy()

    # Scale elimination: divide prices by last bar's close
    last_close = candles[-1, 3]
    if last_close > 0:
        result[:, :4] = candles[:, :4] / last_close

    # Volume normalized by last bar's volume
    last_vol = candles[-1, 4]
    if last_vol > 0:
        result[:, 4] = candles[:, 4] / last_vol
    else:
        result[:, 4] = 0

    return result


N_FEATURES = 5
