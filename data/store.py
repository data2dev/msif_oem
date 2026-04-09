"""
DataStore
=========
Central state for all incoming market data.
Maintains per-symbol:
  - OHLCV candle history (rolling deque)
  - Local order book (for OBI / VW-OBI)
  - OFI delta counters (additions vs cancellations)
  - Trade tick buffer (for TAQ features per bar)

Every minute, the bar_manager snapshots the current state into a completed bar
and resets the intra-bar accumulators.
"""

import time
import logging
import threading
import numpy as np
from collections import deque, defaultdict
import config as cfg

log = logging.getLogger(__name__)


class BarAccumulator:
    """Accumulates ticks within a single 1-minute bar."""

    def __init__(self):
        self.trades = []           # (price, qty, side, ts)
        self.ofi_additions = 0
        self.ofi_cancels = 0
        self.book_snapshots = []   # periodic snapshots for intra-bar analysis

    def reset(self):
        self.trades.clear()
        self.ofi_additions = 0
        self.ofi_cancels = 0
        self.book_snapshots.clear()


class SymbolState:
    """Per-symbol data state."""

    def __init__(self, max_bars: int = 720):
        # Completed OHLCV bars: list of dicts
        self.candles = deque(maxlen=max_bars)

        # Live order book
        self.bids = {}    # price_str -> volume
        self.asks = {}    # price_str -> volume

        # Current bar accumulator
        self.current_bar = BarAccumulator()

        # Completed bar features (for the feature pipeline)
        self.bar_features = deque(maxlen=max_bars)

    def sorted_bids(self) -> list:
        """Sorted bids descending by price. Returns [(price, vol), ...]."""
        return sorted(
            [(float(p), v) for p, v in self.bids.items() if v > 0],
            key=lambda x: -x[0]
        )

    def sorted_asks(self) -> list:
        """Sorted asks ascending by price. Returns [(price, vol), ...]."""
        return sorted(
            [(float(p), v) for p, v in self.asks.items() if v > 0],
            key=lambda x: x[0]
        )


class DataStore:
    """Thread-safe central data store for all market data."""

    def __init__(self):
        self._lock = threading.Lock()
        self.symbols: dict[str, SymbolState] = {}
        self._bar_callbacks = []  # called when a new bar is completed

        for ws_sym in cfg.PAIR_LIST_WS:
            self.symbols[ws_sym] = SymbolState(max_bars=cfg.CANDLE_HISTORY_BARS)

    def on_bar_complete(self, callback):
        """Register a callback(symbol, bar_data) for completed bars."""
        self._bar_callbacks.append(callback)

    # ── Order Book ──────────────────────────

    def book_snapshot(self, symbol: str, bids: list, asks: list):
        """Full book snapshot from WS. bids/asks: [(price, vol), ...]."""
        if symbol not in self.symbols:
            return
        with self._lock:
            state = self.symbols[symbol]
            state.bids = {str(p): v for p, v in bids}
            state.asks = {str(p): v for p, v in asks}
        log.debug(f"{symbol} book snapshot: {len(bids)} bids, {len(asks)} asks")

    def book_update(self, symbol: str, bids: list, asks: list):
        """Incremental book update. Tracks OFI (additions vs cancellations)."""
        if symbol not in self.symbols:
            return
        with self._lock:
            state = self.symbols[symbol]
            bar = state.current_bar

            for price, vol in bids:
                key = str(price)
                if vol == 0:
                    bar.ofi_cancels += 1
                    state.bids.pop(key, None)
                else:
                    if key not in state.bids or state.bids[key] == 0:
                        bar.ofi_additions += 1
                    state.bids[key] = vol

            for price, vol in asks:
                key = str(price)
                if vol == 0:
                    bar.ofi_cancels += 1
                    state.asks.pop(key, None)
                else:
                    if key not in state.asks or state.asks[key] == 0:
                        bar.ofi_additions += 1
                    state.asks[key] = vol

    # ── Trades ──────────────────────────────

    def trade_tick(self, symbol: str, price: float, qty: float,
                   side: str, timestamp: str):
        """Single trade tick from WS."""
        if symbol not in self.symbols:
            return
        with self._lock:
            state = self.symbols[symbol]
            state.current_bar.trades.append((price, qty, side, timestamp))

    # ── Candle Loading (REST bootstrap) ─────

    def load_candles(self, symbol: str, candles: list):
        """Load historical OHLCV from REST.
        candles: list of [ts, o, h, l, c, vwap, vol, count]
        """
        if symbol not in self.symbols:
            return
        with self._lock:
            state = self.symbols[symbol]
            for c in candles:
                bar = {
                    "ts": int(c[0]),
                    "open": float(c[1]),
                    "high": float(c[2]),
                    "low": float(c[3]),
                    "close": float(c[4]),
                    "vwap": float(c[5]),
                    "volume": float(c[6]),
                    "count": int(c[7]),
                }
                state.candles.append(bar)
            log.info(f"{symbol} loaded {len(candles)} historical candles")

    # ── Bar Completion ──────────────────────

    def complete_bar(self, symbol: str, candle: dict):
        """
        Called by the bar manager when a 1-minute bar closes.
        Snapshots intra-bar state, appends candle, resets accumulators.
        
        Args:
            candle: OHLCV dict from REST poll
        """
        if symbol not in self.symbols:
            return
        with self._lock:
            state = self.symbols[symbol]
            bar = state.current_bar

            # Build completed bar data
            bar_data = {
                # OHLCV
                **candle,
                # Book snapshot at bar close
                "bids": state.sorted_bids()[:cfg.BOOK_DEPTH],
                "asks": state.sorted_asks()[:cfg.BOOK_DEPTH],
                # Intra-bar trade aggregates
                "trades": list(bar.trades),
                "n_trades": len(bar.trades),
                "buy_vol": sum(q for _, q, s, _ in bar.trades if s == "buy"),
                "sell_vol": sum(q for _, q, s, _ in bar.trades if s == "sell"),
                # OFI counters
                "ofi_additions": bar.ofi_additions,
                "ofi_cancels": bar.ofi_cancels,
            }

            state.candles.append(candle)
            state.bar_features.append(bar_data)

            # Reset for next bar
            bar.reset()

        # Notify callbacks
        for cb in self._bar_callbacks:
            try:
                cb(symbol, bar_data)
            except Exception as e:
                log.error(f"Bar callback error: {e}", exc_info=True)

    # ── Accessors ───────────────────────────

    def get_candle_array(self, symbol: str, n_bars: int = None) -> np.ndarray:
        """Get OHLCV history as numpy array [bars, 5] (o, h, l, c, v)."""
        if symbol not in self.symbols:
            return np.array([])
        with self._lock:
            candles = list(self.symbols[symbol].candles)
            if n_bars:
                candles = candles[-n_bars:]
        if not candles:
            return np.array([])
        return np.array([
            [c["open"], c["high"], c["low"], c["close"], c["volume"]]
            for c in candles
        ], dtype=np.float64)

    def get_bar_features(self, symbol: str, n_bars: int = None) -> list:
        """Get completed bar feature dicts."""
        if symbol not in self.symbols:
            return []
        with self._lock:
            bars = list(self.symbols[symbol].bar_features)
            if n_bars:
                bars = bars[-n_bars:]
            return bars

    def get_book_state(self, symbol: str) -> tuple:
        """Get current book as (bids, asks) sorted lists."""
        if symbol not in self.symbols:
            return [], []
        with self._lock:
            state = self.symbols[symbol]
            return state.sorted_bids(), state.sorted_asks()
