"""
Historical Backfill
====================
Pulls historical OHLCV + trades from Kraken REST and builds
a training dataset in minutes instead of 24 real-time hours.

Kraken returns ~720 candles per OHLC call. By paginating with
the 'since' parameter, we can pull weeks of 1-minute data.

Limitation: no historical order book, so Category D (microstructure)
features are approximated from trade data only. Once you go live,
the real-time collector fills in the full order book features.

Usage:
    python backfill.py --days 7                  # 7 days of history
    python backfill.py --days 14 --pair XXBTZUSD  # 14 days, BTC only
"""

import argparse
import json
import logging
import os
import time
import numpy as np
from data import kraken_rest as api
from features import ohlcv as ohlcv_mod
from features import intrabar as intrabar_mod
from features import multibar as multibar_mod
from features import microstructure as micro_mod
import config as cfg

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(name)-12s] %(levelname)-7s %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("backfill")

# Kraken rate limit: ~1 req/sec for public endpoints
RATE_LIMIT_DELAY = 1.2


def fetch_candle_history(pair: str, days: int) -> list:
    """
    Fetch N days of 1-minute candles via REST pagination.
    
    Kraken OHLC endpoint returns ~720 bars per call.
    The 'last' field is a NANOSECOND timestamp used as the pagination cursor.
    The 'since' parameter accepts both seconds and nanoseconds.
    We track progress using candle timestamps (seconds) but pass
    Kraken's 'last' token directly as the next 'since' value.
    
    Returns list of [ts, o, h, l, c, vwap, vol, count].
    """
    all_candles = []
    end_ts = int(time.time())
    start_ts = end_ts - (days * 86400)
    current_since = start_ts  # first call uses seconds

    log.info(f"  Fetching {days} days of 1m candles for {pair}")
    log.info(f"  From: {time.strftime('%Y-%m-%d %H:%M', time.localtime(start_ts))}")
    log.info(f"  To:   {time.strftime('%Y-%m-%d %H:%M', time.localtime(end_ts))}")

    calls = 0
    last_candle_ts = start_ts  # track progress in seconds

    while last_candle_ts < end_ts - 120:
        try:
            import requests
            url = f"{cfg.REST_BASE}/0/public/OHLC"
            resp = requests.get(url, params={
                "pair": pair, "interval": cfg.OHLCV_INTERVAL, "since": current_since,
            }, timeout=10)
            resp.raise_for_status()
            body = resp.json()

            if body.get("error"):
                log.error(f"    Kraken error: {body['error']}")
                break

            result = body["result"]
            key = [k for k in result if k != "last"][0]
            candles = result[key]
            pagination_last = result.get("last", 0)  # nanosecond token

            if not candles:
                break

            # Filter to only bars within our range
            new_bars = [c for c in candles if start_ts <= c[0] <= end_ts]
            all_candles.extend(new_bars)

            # Track progress using actual candle timestamps (seconds)
            new_last_ts = candles[-1][0]
            if new_last_ts <= last_candle_ts:
                # No progress
                break
            last_candle_ts = new_last_ts

            # Use Kraken's nanosecond token for next call
            current_since = pagination_last

            calls += 1
            if calls % 5 == 0:
                pct = (last_candle_ts - start_ts) / max(end_ts - start_ts, 1) * 100
                log.info(f"    {len(all_candles)} bars fetched ({pct:.0f}% done, "
                         f"up to {time.strftime('%Y-%m-%d %H:%M', time.localtime(last_candle_ts))})")

            time.sleep(RATE_LIMIT_DELAY)

        except Exception as e:
            log.error(f"    Fetch error: {e}. Retrying in 5s...")
            time.sleep(5)

    # Deduplicate by timestamp
    seen = set()
    unique = []
    for c in all_candles:
        if c[0] not in seen:
            seen.add(c[0])
            unique.append(c)

    unique.sort(key=lambda x: x[0])
    log.info(f"  Total: {len(unique)} unique bars in {calls} API calls")
    return unique


def fetch_trade_history(pair: str, days: int) -> dict:
    """
    Fetch recent trade history and bucket into minute bars.
    
    Note: Kraken's Trades endpoint returns up to 1000 trades per call.
    For high-volume pairs, this may only cover a few minutes.
    We paginate to get as much as feasible within rate limits.
    
    Returns: {bar_ts: [trades]} dict, bucketed by bar interval.
    """
    log.info(f"  Fetching trade history for {pair} (last {days} days)")

    trade_buckets = {}
    end_ts = int(time.time())
    start_ts = end_ts - (days * 86400)
    bar_seconds = cfg.OHLCV_INTERVAL * 60  # bucket size in seconds

    # Kraken trade 'since' uses nanosecond timestamps
    current_since = start_ts * 1_000_000_000
    calls = 0
    max_calls = min(days * 100, 500)  # cap to avoid excessive API usage

    while calls < max_calls:
        try:
            result = api.public("Trades", {
                "pair": pair,
                "since": current_since,
                "count": 1000,
            })
            key = [k for k in result if k != "last"][0]
            trades = result[key]
            last_id = result.get("last", "")

            if not trades:
                break

            for t in trades:
                price, vol, ts, side, otype, misc = t[:6]
                ts_sec = int(float(ts))
                bar_ts = ts_sec - (ts_sec % bar_seconds)

                if bar_ts not in trade_buckets:
                    trade_buckets[bar_ts] = []
                trade_buckets[bar_ts].append({
                    "price": float(price),
                    "vol": float(vol),
                    "side": side,
                })

            # Advance
            if last_id:
                new_since = int(last_id)
                if new_since <= current_since:
                    break
                current_since = new_since
            else:
                break

            # Check if we've reached current time
            last_trade_ts = int(float(trades[-1][2]))
            if last_trade_ts >= end_ts:
                break

            calls += 1
            if calls % 20 == 0:
                log.info(f"    {len(trade_buckets)} bar-buckets covered, {calls} API calls")

            time.sleep(RATE_LIMIT_DELAY)

        except Exception as e:
            log.error(f"    Trade fetch error: {e}. Retrying...")
            time.sleep(5)

    log.info(f"  Trade data: {len(trade_buckets)} bar-buckets from {calls} calls")
    return trade_buckets


def build_bar_data(candles: list, trade_buckets: dict) -> list:
    """
    Build bar_data dicts matching the DataStore format.
    Approximates order book features from trade data where book data is unavailable.
    """
    bars = []

    for c in candles:
        ts = int(c[0])
        o, h, l, close = float(c[1]), float(c[2]), float(c[3]), float(c[4])
        vwap, vol, count = float(c[5]), float(c[6]), int(c[7])

        bar_seconds = cfg.OHLCV_INTERVAL * 60
        bar_ts = ts - (ts % bar_seconds)
        bucket_trades = trade_buckets.get(bar_ts, [])

        buy_vol = sum(t["vol"] for t in bucket_trades if t["side"] == "b")
        sell_vol = sum(t["vol"] for t in bucket_trades if t["side"] == "s")
        n_trades = len(bucket_trades) if bucket_trades else count

        # Approximate order book from trade data
        # Since we don't have historical book snapshots, we synthesize
        # plausible book levels from the bar's price range
        mid = (h + l) / 2
        spread_est = max((h - l) * 0.1, close * 0.0001)  # estimated spread

        bids = []
        asks = []
        for j in range(25):
            bid_price = mid - spread_est * (0.5 + j * 0.5)
            ask_price = mid + spread_est * (0.5 + j * 0.5)
            # Volume decays with distance (approximation)
            level_vol = max(vol * 0.1 / (1 + j * 0.3), 0.001)
            bids.append((bid_price, level_vol))
            asks.append((ask_price, level_vol))

        # Approximate OFI from trade count
        ofi_additions = max(int(n_trades * 0.6), 1)
        ofi_cancels = max(int(n_trades * 0.3), 1)

        trade_list = [
            (t["price"], t["vol"], "buy" if t["side"] == "b" else "sell", "")
            for t in bucket_trades
        ]

        bars.append({
            "ts": ts,
            "open": o, "high": h, "low": l, "close": close,
            "vwap": vwap, "volume": vol, "count": count,
            "bids": bids,
            "asks": asks,
            "trades": trade_list,
            "n_trades": n_trades,
            "buy_vol": buy_vol,
            "sell_vol": sell_vol,
            "ofi_additions": ofi_additions,
            "ofi_cancels": ofi_cancels,
        })

    return bars


def compute_all_features(candles_raw: np.ndarray, bars: list,
                          lookback: int = 30) -> list:
    """
    Compute all four feature categories for each bar.
    Returns list of (features_dict, close_price) tuples.
    """
    n = len(bars)
    if n < lookback + cfg.FORWARD_BARS:
        log.warning(f"Not enough bars: {n} < {lookback + cfg.FORWARD_BARS}")
        return []

    # Pre-compute full multi-bar features
    feat_c_full = multibar_mod.compute(candles_raw)

    samples = []
    for i in range(lookback, n - cfg.FORWARD_BARS):
        window_candles = candles_raw[i - lookback:i]
        window_bars = bars[i - lookback:i]

        feat_a = ohlcv_mod.compute(window_candles)
        feat_b = intrabar_mod.compute(window_bars)
        feat_c = feat_c_full[i - lookback:i]
        feat_d = micro_mod.compute(window_bars)

        # Forward return label
        close_now = bars[i]["close"]
        close_future = bars[i + cfg.FORWARD_BARS]["close"]
        if close_now <= 0:
            continue
        label = (close_future - close_now) / close_now

        features = {
            "A": feat_a.tolist(),
            "B": feat_b.tolist(),
            "C": feat_c.tolist(),
            "D": feat_d.tolist(),
        }

        samples.append({
            "symbol": "",  # filled in by caller
            "ts": bars[i]["ts"],
            "features": features,
            "close": close_now,
            "label": label,
        })

    return samples


def main():
    parser = argparse.ArgumentParser(description="MSIF-OEM Historical Backfill")
    parser.add_argument("--days", type=int, default=7, help="Days of history (default: 7)")
    parser.add_argument("--pair", type=str, default=None, help="Single pair (default: all)")
    parser.add_argument("--skip-trades", action="store_true",
                        help="Skip trade fetch (faster, less accurate micro features)")
    parser.add_argument("--dir", type=str, default=cfg.DATA_DIR, help="Output directory")
    args = parser.parse_args()

    os.makedirs(args.dir, exist_ok=True)

    pairs = {args.pair: cfg.PAIRS.get(args.pair, args.pair)} if args.pair else cfg.PAIRS

    log.info("=" * 60)
    log.info("  MSIF-OEM Historical Backfill")
    log.info(f"  Days: {args.days}")
    log.info(f"  Pairs: {list(pairs.keys())}")
    log.info(f"  Expected bars per pair: ~{args.days * 1440}")
    est_minutes = len(pairs) * (args.days * 2 + 5)
    log.info(f"  Estimated time: ~{est_minutes} minutes")
    log.info("=" * 60)

    all_samples = []

    for rest_pair, ws_sym in pairs.items():
        log.info(f"\n{'─' * 40}")
        log.info(f"Processing {ws_sym} ({rest_pair})")
        log.info(f"{'─' * 40}")

        # Fetch candles
        candles = fetch_candle_history(rest_pair, args.days)
        if len(candles) < cfg.LOOKBACK_BARS + cfg.FORWARD_BARS + 10:
            log.warning(f"Insufficient candles for {ws_sym}: {len(candles)}")
            continue

        # Fetch trades (optional)
        if not args.skip_trades:
            trade_buckets = fetch_trade_history(rest_pair, args.days)
        else:
            trade_buckets = {}
            log.info("  Skipping trade fetch (--skip-trades)")

        # Build bar data
        candles_raw = np.array([
            [float(c[1]), float(c[2]), float(c[3]), float(c[4]), float(c[6])]
            for c in candles
        ], dtype=np.float64)

        bars = build_bar_data(candles, trade_buckets)
        log.info(f"  Built {len(bars)} bar records")

        # Compute features
        log.info(f"  Computing features (lookback={cfg.LOOKBACK_BARS})...")
        samples = compute_all_features(candles_raw, bars)

        for s in samples:
            s["symbol"] = ws_sym

        all_samples.extend(samples)
        log.info(f"  Generated {len(samples)} training samples for {ws_sym}")

    # Save
    if not all_samples:
        log.error("No samples generated. Check data availability.")
        return

    output_path = os.path.join(args.dir, f"backfill_{args.days}d_{int(time.time())}.json")
    save_data = {
        "n_samples": len(all_samples),
        "days": args.days,
        "pairs": list(pairs.values()),
        "type": "backfill",
        "samples": all_samples,
    }

    with open(output_path, "w") as f:
        json.dump(save_data, f)

    log.info(f"\n{'=' * 60}")
    log.info(f"  BACKFILL COMPLETE")
    log.info(f"  Total samples: {len(all_samples)}")
    log.info(f"  Saved to: {output_path}")
    file_mb = os.path.getsize(output_path) / (1024 * 1024)
    log.info(f"  File size: {file_mb:.1f} MB")
    log.info(f"{'=' * 60}")
    log.info(f"\nNext step: python train.py --data {args.dir}")


if __name__ == "__main__":
    main()
