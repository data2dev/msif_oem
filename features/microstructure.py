"""
Category D: Microstructure Factors
===================================
Order book and trade-flow features with anti-spoofing defenses.
Implements all three defenses from the OBI note:
  - Multi-level OBI (3 slices)
  - Volume-weighted OBI (VW-OBI)
  - Order Flow Imbalance (OFI)

Features (14):
  0:  obi_near     - OBI levels 1-3 (immediate spread)
  1:  obi_mid      - OBI levels 4-10
  2:  obi_deep     - OBI levels 11-25
  3:  vw_obi       - volume-weighted OBI (distance-discounted)
  4:  ofi_net      - order flow imbalance (additions - cancels), normalized
  5:  spread       - bid-ask spread / mid price (bps)
  6:  spread_imb   - (ask_depth_near - bid_depth_near) / total_near
  7:  depth_ratio  - total_bid_depth / total_ask_depth
  8:  trade_imb    - (buy_vol - sell_vol) / total_vol
  9:  large_trade  - fraction of volume from trades > 2x median size
  10: vpin         - volume-sync'd probability of informed trading (simplified)
  11: cancel_rate  - ofi_cancels / (ofi_additions + ofi_cancels)
  12: obi_diverge  - obi_near - obi_deep (spoofing signature)
  13: book_pressure - weighted bid pressure vs ask pressure
"""

import numpy as np
import config as cfg


def compute(bars: list) -> np.ndarray:
    """
    Compute microstructure factors from completed bar data.
    
    Args:
        bars: list of bar_data dicts containing bids, asks, trades, ofi counters
        
    Returns:
        [n_bars, 14] feature array
    """
    n = len(bars)
    if n == 0:
        return np.zeros((0, N_FEATURES))

    features = np.zeros((n, N_FEATURES))

    for i, bar in enumerate(bars):
        bids = bar.get("bids", [])  # [(price, vol), ...] sorted desc
        asks = bar.get("asks", [])  # [(price, vol), ...] sorted asc

        if not bids or not asks:
            continue

        mid = (bids[0][0] + asks[0][0]) / 2 if bids and asks else 0
        if mid <= 0:
            continue

        # ── DEFENSE A: Multi-level OBI ──
        near_s, near_e = cfg.OBI_NEAR_LEVELS
        mid_s, mid_e = cfg.OBI_MID_LEVELS
        deep_s, deep_e = cfg.OBI_DEEP_LEVELS

        features[i, 0] = _obi_slice(bids, asks, near_s, near_e)
        features[i, 1] = _obi_slice(bids, asks, mid_s, mid_e)
        features[i, 2] = _obi_slice(bids, asks, deep_s, deep_e)

        # ── DEFENSE B: Volume-Weighted OBI ──
        features[i, 3] = _vw_obi(bids, asks, mid)

        # ── DEFENSE C: Order Flow Imbalance ──
        ofi_add = bar.get("ofi_additions", 0)
        ofi_cancel = bar.get("ofi_cancels", 0)
        ofi_total = ofi_add + ofi_cancel
        if ofi_total > 0:
            features[i, 4] = (ofi_add - ofi_cancel) / ofi_total
        else:
            features[i, 4] = 0

        # ── Spread ──
        spread = asks[0][0] - bids[0][0]
        features[i, 5] = (spread / mid) * 10000  # in bps

        # ── Spread imbalance (near levels) ──
        near_bid_depth = sum(v for _, v in bids[:3])
        near_ask_depth = sum(v for _, v in asks[:3])
        near_total = near_bid_depth + near_ask_depth
        if near_total > 0:
            features[i, 6] = (near_ask_depth - near_bid_depth) / near_total

        # ── Depth ratio ──
        total_bid = sum(v for _, v in bids)
        total_ask = sum(v for _, v in asks)
        if total_ask > 0:
            features[i, 7] = total_bid / total_ask
        else:
            features[i, 7] = 1.0

        # ── Trade imbalance ──
        buy_vol = bar.get("buy_vol", 0)
        sell_vol = bar.get("sell_vol", 0)
        total_trade = buy_vol + sell_vol
        if total_trade > 0:
            features[i, 8] = (buy_vol - sell_vol) / total_trade

        # ── Large trade fraction ──
        trades = bar.get("trades", [])
        if len(trades) > 2:
            sizes = [t[1] for t in trades]
            median_size = np.median(sizes)
            if median_size > 0:
                large = sum(s for s in sizes if s > 2 * median_size)
                features[i, 9] = large / sum(sizes)

        # ── Simplified VPIN ──
        # Approximate: |buy_vol - sell_vol| / total_vol over rolling window
        if total_trade > 0:
            features[i, 10] = abs(buy_vol - sell_vol) / total_trade

        # ── Cancel rate ──
        if ofi_total > 0:
            features[i, 11] = ofi_cancel / ofi_total

        # ── OBI divergence (spoofing signature) ──
        features[i, 12] = features[i, 0] - features[i, 2]

        # ── Weighted book pressure ──
        features[i, 13] = _book_pressure(bids, asks, mid)

    features = np.nan_to_num(features, nan=0.0, posinf=3.0, neginf=-3.0)
    return features


def _obi_slice(bids: list, asks: list, start: int, end: int) -> float:
    """OBI for a specific depth slice."""
    bid_vol = sum(v for _, v in bids[start:end])
    ask_vol = sum(v for _, v in asks[start:end])
    total = bid_vol + ask_vol
    if total == 0:
        return 0.0
    return (bid_vol - ask_vol) / total


def _vw_obi(bids: list, asks: list, mid: float) -> float:
    """Volume-weighted OBI: discount volume by distance from mid."""
    bid_weighted = 0.0
    ask_weighted = 0.0

    for price, vol in bids:
        dist = abs(price - mid) / mid
        weight = 1.0 / (1.0 + 100 * dist)  # decay factor
        bid_weighted += vol * weight

    for price, vol in asks:
        dist = abs(price - mid) / mid
        weight = 1.0 / (1.0 + 100 * dist)
        ask_weighted += vol * weight

    total = bid_weighted + ask_weighted
    if total == 0:
        return 0.0
    return (bid_weighted - ask_weighted) / total


def _book_pressure(bids: list, asks: list, mid: float) -> float:
    """Weighted pressure: sum(vol/distance) for bids vs asks."""
    bid_pressure = 0.0
    ask_pressure = 0.0

    for price, vol in bids[:10]:
        dist = max(mid - price, 1e-8)
        bid_pressure += vol / dist

    for price, vol in asks[:10]:
        dist = max(price - mid, 1e-8)
        ask_pressure += vol / dist

    total = bid_pressure + ask_pressure
    if total == 0:
        return 0.0
    return (bid_pressure - ask_pressure) / total


N_FEATURES = 14
