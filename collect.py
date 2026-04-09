"""
Data Collector
==============
Phase 1: Collect data before training.

Runs the data ingestion pipeline (REST + WebSocket) and saves
completed bar features to disk for later training.

Usage:
    python collect.py --hours 24    # collect for 24 hours
    python collect.py --bars 1000   # collect 1000 bars then stop
"""

import argparse
import asyncio
import json
import logging
import os
import signal
import time
import numpy as np
from data.store import DataStore
from data.kraken_ws import KrakenWebSocket
from data import kraken_rest as api
from features.pipeline import FeaturePipeline
import config as cfg

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(name)-12s] %(levelname)-7s %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("collector")


class Collector:
    def __init__(self, save_dir: str):
        self.save_dir = save_dir
        os.makedirs(save_dir, exist_ok=True)

        self.store = DataStore()
        self.ws_client = KrakenWebSocket(self.store)
        self.pipeline = FeaturePipeline(self.store)

        self.collected_samples = []  # list of (symbol, features_dict, label_or_None)
        self.bar_count = 0
        self._stop = False

    async def run(self, max_hours: float = None, max_bars: int = None):
        """Main collection loop."""
        log.info("=" * 60)
        log.info("  MSIF-OEM Data Collector")
        log.info(f"  Pairs: {cfg.PAIR_LIST_WS}")
        log.info(f"  Save dir: {self.save_dir}")
        if max_hours:
            log.info(f"  Duration: {max_hours} hours")
        if max_bars:
            log.info(f"  Target: {max_bars} bars")
        log.info("=" * 60)

        # Bootstrap: load historical candles via REST
        await self._bootstrap()

        # Start WebSocket streaming
        ws_task = asyncio.create_task(self.ws_client.start())

        # Wait for WS connection
        await asyncio.sleep(3)

        # Bar polling loop
        start_time = time.time()
        try:
            while not self._stop:
                await self._poll_bars()

                if max_hours and (time.time() - start_time) > max_hours * 3600:
                    log.info(f"Time limit reached ({max_hours}h)")
                    break
                if max_bars and self.bar_count >= max_bars:
                    log.info(f"Bar limit reached ({max_bars})")
                    break

                # Wait until next minute
                now = time.time()
                sleep_sec = 60 - (now % 60) + 2  # 2s after minute boundary
                await asyncio.sleep(sleep_sec)

        except asyncio.CancelledError:
            pass
        finally:
            await self.ws_client.stop()
            ws_task.cancel()
            self._save_data()

    async def _bootstrap(self):
        """Load recent candle history from REST."""
        log.info("Bootstrapping candle history from REST...")

        for rest_pair, ws_sym in cfg.PAIRS.items():
            try:
                candles = api.get_ohlcv(rest_pair, interval=cfg.OHLCV_INTERVAL)
                # Map REST pair to WS symbol for store
                self.store.load_candles(ws_sym, candles[:-1])  # exclude forming bar
                log.info(f"  {ws_sym}: loaded {len(candles)-1} candles")
            except Exception as e:
                log.error(f"  {ws_sym}: bootstrap failed: {e}")

    async def _poll_bars(self):
        """Poll REST for latest completed candles and trigger bar completion."""
        for rest_pair, ws_sym in cfg.PAIRS.items():
            try:
                candles = api.get_ohlcv(rest_pair, interval=cfg.OHLCV_INTERVAL)
                if len(candles) < 2:
                    continue

                # Second-to-last is the latest completed bar
                latest = candles[-2]
                bar = {
                    "ts": int(latest[0]),
                    "open": float(latest[1]),
                    "high": float(latest[2]),
                    "low": float(latest[3]),
                    "close": float(latest[4]),
                    "vwap": float(latest[5]),
                    "volume": float(latest[6]),
                    "count": int(latest[7]),
                }

                # Check if this is a new bar
                existing = self.store.get_candle_array(ws_sym, n_bars=1)
                if existing.shape[0] > 0:
                    last_close = existing[-1, 3]
                    last_ts = bar["ts"]
                    # Simple dedup: skip if same timestamp
                    bars_list = list(self.store.symbols[ws_sym].candles)
                    if bars_list and bars_list[-1].get("ts") == bar["ts"]:
                        continue

                self.store.complete_bar(ws_sym, bar)
                self.bar_count += 1

                # Compute features if enough data
                feats = self.pipeline.compute_features(ws_sym)
                if feats is not None:
                    self.collected_samples.append({
                        "symbol": ws_sym,
                        "ts": bar["ts"],
                        "features": {k: v.tolist() for k, v in feats.items()},
                        "close": bar["close"],
                    })

                if self.bar_count % 10 == 0:
                    log.info(f"Bars collected: {self.bar_count}, "
                             f"samples: {len(self.collected_samples)}")

            except Exception as e:
                log.error(f"Poll error for {ws_sym}: {e}")

    def _save_data(self):
        """Save collected data to disk."""
        if not self.collected_samples:
            log.warning("No samples collected")
            return

        path = os.path.join(self.save_dir, f"collected_{int(time.time())}.json")

        # Convert to serializable format
        save_data = {
            "n_samples": len(self.collected_samples),
            "n_bars": self.bar_count,
            "pairs": cfg.PAIR_LIST_WS,
            "samples": self.collected_samples,
        }

        with open(path, "w") as f:
            json.dump(save_data, f)

        log.info(f"Saved {len(self.collected_samples)} samples to {path}")
        log.info(f"Total bars collected: {self.bar_count}")


def main():
    parser = argparse.ArgumentParser(description="MSIF-OEM Data Collector")
    parser.add_argument("--hours", type=float, default=None, help="Collection duration in hours")
    parser.add_argument("--bars", type=int, default=None, help="Number of bars to collect")
    parser.add_argument("--dir", type=str, default=cfg.DATA_DIR, help="Save directory")
    args = parser.parse_args()

    if args.hours is None and args.bars is None:
        args.hours = 12  # default: 12 hours
        log.info("No duration specified, defaulting to 12 hours")

    collector = Collector(save_dir=args.dir)

    # Handle graceful shutdown
    loop = asyncio.new_event_loop()

    def shutdown(sig, frame):
        log.info("Shutdown signal received")
        collector._stop = True

    signal.signal(signal.SIGINT, shutdown)
    signal.signal(signal.SIGTERM, shutdown)

    loop.run_until_complete(collector.run(max_hours=args.hours, max_bars=args.bars))


if __name__ == "__main__":
    main()
