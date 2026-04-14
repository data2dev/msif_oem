"""
Feature Pipeline
================
Orchestrates all four feature categories into the tensor format
expected by the PTE-TFE network.

Produces 4 tensors of shape [lookback, n_features_per_category]
for each symbol at each bar.
"""

import numpy as np
import logging
from features import ohlcv, intrabar, multibar, microstructure
import config as cfg

log = logging.getLogger(__name__)


class FeaturePipeline:
    """Compute and cache features for all symbols."""

    def __init__(self, store):
        self.store = store

        # Feature dimensions
        self.dims = {
            "A": ohlcv.N_FEATURES,          # 5
            "B": intrabar.N_FEATURES,        # 10
            "C": multibar.N_FEATURES,        # 24
            "D": microstructure.N_FEATURES,  # 14
        }
        self.total_dim = sum(self.dims.values())
        log.info(f"Feature dims: A={self.dims['A']}, B={self.dims['B']}, "
                 f"C={self.dims['C']}, D={self.dims['D']}, total={self.total_dim}")

    def compute_features(self, symbol: str) -> dict | None:
        """
        Compute all four feature categories for a symbol.
        
        Returns dict with keys 'A', 'B', 'C', 'D', each [lookback, n_features],
        or None if insufficient data.
        """
        bars = self.store.get_bar_features(symbol)
        candles_raw = self.store.get_candle_array(symbol)

        lb = cfg.LOOKBACK_BARS  # 30

        if len(bars) < lb or candles_raw.shape[0] < lb:
            return None

        # Take last `lb` bars for each category
        bars_window = bars[-lb:]
        candles_window = candles_raw[-lb:]

        try:
            # Category A: Raw OHLCV (scale-eliminated)
            feat_a = ohlcv.compute(candles_window)

            # Category B: Intra-bar PV
            feat_b = intrabar.compute(bars_window)

            # Category C: Multi-bar PV (needs more history for lookback)
            # Use full candle history for rolling computations, then slice
            feat_c_full = multibar.compute(candles_raw)
            feat_c = feat_c_full[-lb:]

            # Category D: Microstructure
            feat_d = microstructure.compute(bars_window)

            # Validate shapes
            assert feat_a.shape == (lb, self.dims["A"]), f"A shape {feat_a.shape}"
            assert feat_b.shape == (lb, self.dims["B"]), f"B shape {feat_b.shape}"
            assert feat_c.shape == (lb, self.dims["C"]), f"C shape {feat_c.shape}"
            assert feat_d.shape == (lb, self.dims["D"]), f"D shape {feat_d.shape}"

            return {"A": feat_a, "B": feat_b, "C": feat_c, "D": feat_d}

        except Exception as e:
            log.error(f"Feature computation failed for {symbol}: {e}", exc_info=True)
            return None

    def compute_batch(self) -> dict:
        """
        Compute features for all symbols.
        
        Returns:
            {symbol: {A, B, C, D}} for symbols with sufficient data
        """
        batch = {}
        for ws_sym in cfg.PAIR_LIST_WS:
            feats = self.compute_features(ws_sym)
            if feats is not None:
                batch[ws_sym] = feats
        return batch

    def features_to_tensors(self, batch: dict) -> tuple:
        """
        Convert a batch of features to numpy arrays for the network.
        
        Args:
            batch: {symbol: {A, B, C, D}} from compute_batch
            
        Returns:
            (tensor_A, tensor_B, tensor_C, tensor_D, symbols)
            Each tensor: [n_symbols, lookback, n_features]
            symbols: list of symbol names in same order
        """
        if not batch:
            return None, None, None, None, []

        symbols = list(batch.keys())
        n = len(symbols)
        lb = cfg.LOOKBACK_BARS

        ta = np.zeros((n, lb, self.dims["A"]))
        tb = np.zeros((n, lb, self.dims["B"]))
        tc = np.zeros((n, lb, self.dims["C"]))
        td = np.zeros((n, lb, self.dims["D"]))

        for i, sym in enumerate(symbols):
            ta[i] = batch[sym]["A"]
            tb[i] = batch[sym]["B"]
            tc[i] = batch[sym]["C"]
            td[i] = batch[sym]["D"]

        return ta, tb, tc, td, symbols

    def compute_labels(self, symbol: str, offset: int = None) -> np.ndarray | None:
        """
        Compute forward return labels for training.
        
        Args:
            symbol: trading pair
            offset: if set, compute label at this bar index
            
        Returns:
            forward_return float, or None if not enough future data
        """
        candles = self.store.get_candle_array(symbol)
        if candles.shape[0] < cfg.LOOKBACK_BARS + cfg.FORWARD_BARS:
            return None

        if offset is None:
            # Label for the last complete window
            idx = candles.shape[0] - cfg.FORWARD_BARS - 1
        else:
            idx = offset

        if idx < 0 or idx + cfg.FORWARD_BARS >= candles.shape[0]:
            return None

        price_now = candles[idx, 3]  # close
        price_future = candles[idx + cfg.FORWARD_BARS, 3]

        if price_now <= 0:
            return None

        return (price_future - price_now) / price_now
