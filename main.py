"""
MSIF-OEM Main Loop (Self-Bootstrapping)
=========================================
Automatically handles the full lifecycle:
  1. Checks for trained models
  2. If missing: runs backfill + training automatically
  3. Starts live/paper trading with RL agent

Usage:
    python main.py                    # paper trading (default)
    python main.py --live             # live trading
    python main.py --no-rl            # disable RL, use simple signal threshold
    python main.py --backfill-days 14 # override backfill duration (default 7)
    python main.py --retrain          # force retrain even if models exist
"""

import argparse
import asyncio
import glob
import logging
import os
import signal
import subprocess
import sys
import time
import numpy as np
import torch
import config as cfg
from data.store import DataStore
from data.kraken_ws import KrakenWebSocket
from data import kraken_rest as api
from features.pipeline import FeaturePipeline
from model.transformer import PTETFE
from model.esgd import ESGDModel
from trading.signals import SignalGenerator
from trading.risk import RiskManager
from trading.executor import Executor
from rl.state import StateBuilder
from rl.agent import SACAgent
from rl.reward import RewardCalculator
from rl.outcome_tracker import OutcomeTracker

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(name)-12s] %(levelname)-7s %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("main")


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  AUTO-BOOTSTRAP: backfill + train if models don't exist
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def ensure_models_exist(model_dir: str, data_dir: str, backfill_days: int,
                        force_retrain: bool = False):
    """
    Check for trained models. If missing (or force_retrain), run
    backfill and training automatically.
    """
    ptetfe_path = os.path.join(model_dir, "ptetfe.pt")
    esgd_path = os.path.join(model_dir, "esgd.pkl")

    models_exist = os.path.exists(ptetfe_path) and os.path.exists(esgd_path)

    if models_exist and not force_retrain:
        log.info(f"Models found: {ptetfe_path}, {esgd_path}")
        return

    if force_retrain:
        log.info("Force retrain requested")
    else:
        log.info("No trained models found — running automatic setup")

    # ── Step 1: Check for existing training data ──
    os.makedirs(data_dir, exist_ok=True)
    data_files = glob.glob(os.path.join(data_dir, "*.json"))

    if data_files and not force_retrain:
        log.info(f"Found existing data: {len(data_files)} file(s) in {data_dir}")
        log.info("Skipping backfill, using existing data for training")
    else:
        # ── Step 2: Run backfill ──
        log.info("=" * 60)
        log.info(f"  BACKFILL: Pulling {backfill_days} days of history")
        log.info(f"  Estimated time: ~{backfill_days * 2 + 5} minutes")
        log.info("=" * 60)

        result = subprocess.run(
            [sys.executable, "backfill.py",
             "--days", str(backfill_days),
             "--dir", data_dir],
            cwd=os.path.dirname(os.path.abspath(__file__)),
        )
        if result.returncode != 0:
            raise RuntimeError(
                f"Backfill failed with exit code {result.returncode}. "
                f"Check your network connection and Kraken API access."
            )

        # Verify data was created
        data_files = glob.glob(os.path.join(data_dir, "*.json"))
        if not data_files:
            raise RuntimeError("Backfill produced no data files")

        log.info(f"Backfill complete: {len(data_files)} data file(s)")

    # ── Step 3: Run training ──
    log.info("=" * 60)
    log.info("  TRAINING: Building PTE-TFE + ESGD models")
    log.info("  Estimated time: 2-5 minutes")
    log.info("=" * 60)

    result = subprocess.run(
        [sys.executable, "train.py", "--data", data_dir],
        cwd=os.path.dirname(os.path.abspath(__file__)),
    )
    if result.returncode != 0:
        raise RuntimeError(
            f"Training failed with exit code {result.returncode}. "
            f"Check that you have enough data (need 100+ samples)."
        )

    # Verify models were created
    if not os.path.exists(ptetfe_path) or not os.path.exists(esgd_path):
        raise RuntimeError("Training completed but model files not found")

    log.info("Setup complete — models ready")


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  TRADING ENGINE
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

class TradingEngine:

    def __init__(self, model_dir: str, paper_trading: bool = True,
                 use_rl: bool = True):
        self.model_dir = model_dir
        self._stop = False
        self.use_rl = use_rl

        # Core components
        self.store = DataStore()
        self.ws_client = KrakenWebSocket(self.store)
        self.pipeline = FeaturePipeline(self.store)
        self.signal_gen = SignalGenerator()
        self.risk_mgr = RiskManager()
        self.executor = Executor(paper_trading=paper_trading)

        # Models
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.ptetfe = None
        self.esgd = None

        # RL components
        self.n_assets = len(cfg.PAIRS)
        self.symbols_ordered = cfg.PAIR_LIST_WS
        self.state_builder = None
        self.rl_agent = None
        self.reward_calc = None
        self.outcome_tracker = None
        self.last_state = None
        self.last_action = None
        self.last_positions_snapshot = None

        # Delayed label buffer
        self.pending_labels = []

        # Stats
        self.bars_processed = 0
        self.signals_generated = 0
        self.realized_pnl = 0.0
        self.starting_equity = None

    def _load_models(self):
        ptetfe_path = os.path.join(self.model_dir, "ptetfe.pt")
        esgd_path = os.path.join(self.model_dir, "esgd.pkl")

        input_dims = {
            "A": cfg.N_OHLCV_FEATURES,
            "B": cfg.N_INTRABAR_FEATURES,
            "C": cfg.N_MULTIBAR_FEATURES,
            "D": cfg.N_MICRO_FEATURES,
        }
        self.ptetfe = PTETFE(input_dims).to(self.device)
        self.ptetfe.load_state_dict(
            torch.load(ptetfe_path, map_location=self.device, weights_only=True)
        )
        self.ptetfe.eval()
        log.info(f"PTE-TFE loaded ({sum(p.numel() for p in self.ptetfe.parameters()):,} params)")

        self.esgd = ESGDModel()
        self.esgd.load(esgd_path)

        if self.use_rl:
            self.state_builder = StateBuilder(self.n_assets)
            self.rl_agent = SACAgent(
                state_dim=self.state_builder.state_dim,
                action_dim=self.n_assets,
                hidden_dim=256, lr=3e-4, gamma=0.99, tau=0.005,
                buffer_capacity=100_000, batch_size=256,
                warmup_steps=500, device=str(self.device),
            )
            self.reward_calc = RewardCalculator()
            self.outcome_tracker = OutcomeTracker(
                self.rl_agent.replay_buffer,
                self.state_builder,
                self.reward_calc,
            )
            rl_dir = os.path.join(self.model_dir, "rl")
            self.rl_agent.load(rl_dir)
            log.info("RL agent initialized")

    async def start(self):
        mode = "PAPER" if self.executor.paper_trading else "LIVE"
        rl_mode = "RL" if self.use_rl else "RULES"

        log.info("=" * 60)
        log.info(f"  MSIF-OEM Trading Engine [{mode}] [{rl_mode}]")
        log.info(f"  Pairs: {self.symbols_ordered}")
        log.info(f"  Device: {self.device}")
        log.info("=" * 60)

        self._load_models()
        await self._bootstrap()

        ws_task = asyncio.create_task(self.ws_client.start())
        await asyncio.sleep(3)

        try:
            while not self._stop:
                await self._tick()
                now = time.time()
                sleep_sec = 60 - (now % 60) + 2
                await asyncio.sleep(sleep_sec)
        except asyncio.CancelledError:
            pass
        finally:
            await self.ws_client.stop()
            ws_task.cancel()
            self._save_and_report()

    async def _bootstrap(self):
        log.info("Bootstrapping candle history...")
        for rest_pair, ws_sym in cfg.PAIRS.items():
            try:
                candles = api.get_ohlcv(rest_pair, interval=cfg.OHLCV_INTERVAL)
                self.store.load_candles(ws_sym, candles[:-1])
                log.info(f"  {ws_sym}: {len(candles)-1} bars loaded")
            except Exception as e:
                log.error(f"  {ws_sym}: bootstrap failed: {e}")

    async def _tick(self):
        # 1. Poll candles
        for rest_pair, ws_sym in cfg.PAIRS.items():
            try:
                candles = api.get_ohlcv(rest_pair, interval=cfg.OHLCV_INTERVAL)
                if len(candles) < 2:
                    continue
                latest = candles[-2]
                bar = {
                    "ts": int(latest[0]),
                    "open": float(latest[1]), "high": float(latest[2]),
                    "low": float(latest[3]), "close": float(latest[4]),
                    "vwap": float(latest[5]), "volume": float(latest[6]),
                    "count": int(latest[7]),
                }
                sym_candles = list(self.store.symbols[ws_sym].candles)
                if sym_candles and sym_candles[-1].get("ts") == bar["ts"]:
                    continue
                self.store.complete_bar(ws_sym, bar)
                self.bars_processed += 1
            except Exception as e:
                log.error(f"Poll {ws_sym}: {e}")

        # 2. Features → alpha
        batch = self.pipeline.compute_batch()
        if not batch:
            return

        ta, tb, tc, td, symbols = self.pipeline.features_to_tensors(batch)
        if ta is None:
            return

        with torch.no_grad():
            xa = torch.FloatTensor(ta).to(self.device)
            xb = torch.FloatTensor(tb).to(self.device)
            xc = torch.FloatTensor(tc).to(self.device)
            xd = torch.FloatTensor(td).to(self.device)
            features_out = self.ptetfe(xa, xb, xc, xd).cpu().numpy()

        predictions, confidence = self.esgd.predict_with_confidence(features_out)
        pred_dict = {sym: pred for sym, pred in zip(symbols, predictions)}
        conf_dict = {sym: confidence for sym in symbols}
        signals = self.signal_gen.generate(pred_dict, conf_dict)

        # Get prices and spreads
        prices, spreads = {}, {}
        for rest_pair, ws_sym in cfg.PAIRS.items():
            try:
                t = api.get_ticker(rest_pair)
                prices[ws_sym] = t["last"]
                spreads[ws_sym] = ((t["ask"] - t["bid"]) / t["last"]) * 10000
            except Exception:
                pass

        try:
            tb_data = api.get_trade_balance()
            equity = tb_data.get("eb", 0)
        except Exception:
            equity = 0

        if equity <= 0:
            return

        # 3. Execute
        if self.use_rl:
            self._rl_execution(signals, symbols, prices, spreads, equity,
                               features_out)
        else:
            self._rule_execution(signals, symbols, prices, equity)

        # 4. ESGD online update
        for i, sym in enumerate(symbols):
            bars = self.store.get_bar_features(sym, n_bars=1)
            close = bars[-1].get("close", 0) if bars else 0
            self.pending_labels.append((features_out[i], close, time.time(), sym))
        self._process_delayed_labels()

        # 5. Outcome timeouts
        if self.use_rl and self.outcome_tracker:
            self.outcome_tracker.check_timeouts(prices)

        if self.bars_processed % 5 == 0:
            pnl = self._compute_pnl(prices)
            if self.starting_equity is None and equity > 0:
                self.starting_equity = equity
            total_return = ((pnl["total"]) / self.starting_equity * 100) if self.starting_equity else 0

            # Per-asset P&L string
            asset_pnl = " | ".join(
                f"{sym.split('/')[0]}:{d['pnl']:+.2f}({d['pnl_pct']:+.1f}%)"
                for sym, d in pnl["per_asset"].items()
            ) if pnl["per_asset"] else "no positions"

            stats = ""
            if self.use_rl and self.outcome_tracker:
                os_stats = self.outcome_tracker.get_stats()
                stats = (f" | RL steps={self.rl_agent.training_steps}"
                         f" | fills={os_stats['total_fills']}"
                         f" | slip={os_stats['avg_slippage_pct']:.3f}%")
            log.info(
                f"Bars={self.bars_processed} | Signals={self.signals_generated}"
                f" | Pos={len(self.risk_mgr.positions)}"
                f" | ESGD outliers={self.esgd.outlier_count}/{cfg.C_MAX}"
                f"{stats}"
            )
            log.info(
                f"  P&L: realized=${pnl['realized']:+.2f} "
                f"unrealized=${pnl['unrealized']:+.2f} "
                f"total=${pnl['total']:+.2f} ({total_return:+.2f}%) "
                f"| {asset_pnl}"
            )

    def _rl_execution(self, signals, symbols, prices, spreads, equity,
                      features_out):
        mean_feat = features_out.mean(axis=0, keepdims=True)
        cluster_id = int(self.esgd.kmeans.predict(mean_feat)[0])

        state = self.state_builder.build(
            signals=signals, symbols=self.symbols_ordered,
            risk_mgr=self.risk_mgr, spreads=spreads,
            equity=equity, esgd_cluster=cluster_id,
        )

        # Reward for previous step
        if self.last_state is not None and self.last_action is not None:
            prev_positions = self.last_positions_snapshot or {}
            curr_positions = self.risk_mgr.get_positions_snapshot()
            expected_reward = self.reward_calc.compute_expected(
                prev_positions=prev_positions, new_positions=curr_positions,
                prices=prices, equity=equity,
                daily_pnl=self.risk_mgr.daily_pnl,
                starting_equity=self.state_builder.starting_equity or equity,
            )
            transition_idx = self.rl_agent.step(
                self.last_state, self.last_action, expected_reward, state, done=False
            )
            expected_fills = self._build_expected_fills(
                self.last_action, self.symbols_ordered, prices
            )
            self.outcome_tracker.register_action(
                transition_idx=transition_idx,
                expected_fills=expected_fills,
                expected_reward=expected_reward,
                action=self.last_action, state=self.last_state,
                prev_positions=prev_positions, equity=equity,
            )

        action = self.rl_agent.select_action(state)
        targets = self.rl_agent.action_to_positions(
            action, self.symbols_ordered, equity, prices
        )
        targets = self.risk_mgr.validate_rl_targets(targets, equity, prices)

        for ws_sym, target in targets.items():
            rest_pair = self._ws_to_rest(ws_sym)
            if not rest_pair or ws_sym not in prices:
                continue

            current_price = prices[ws_sym]
            current_pos = self.risk_mgr.get_position(ws_sym)

            if self.risk_mgr.check_stop_loss(ws_sym, current_price):
                if current_pos:
                    self._record_close_pnl(ws_sym, current_price)
                    self.executor.close_position(
                        rest_pair, ws_sym, current_pos["side"],
                        current_pos["size"], current_price
                    )
                    self.risk_mgr.close_position(ws_sym)
                continue

            if target["side"] == "flat":
                if current_pos:
                    self._record_close_pnl(ws_sym, current_price)
                    self.executor.close_position(
                        rest_pair, ws_sym, current_pos["side"],
                        current_pos["size"], current_price
                    )
                    self.risk_mgr.close_position(ws_sym)
                continue

            target_size = target["target_size"]

            if not current_pos:
                if target_size > 0:
                    sig = {"direction": target["side"],
                           "strength": abs(target["target_frac"]),
                           "confidence": 1.0}
                    result = self.executor.execute_signal(
                        rest_pair, ws_sym, sig, target_size, current_price
                    )
                    if result and result.get("status") in ("paper_filled", "submitted"):
                        self.risk_mgr.register_position(
                            ws_sym, target["side"], target_size, current_price
                        )
                        self.signals_generated += 1
            else:
                if target["side"] != current_pos["side"]:
                    self._record_close_pnl(ws_sym, current_price)
                    self.executor.close_position(
                        rest_pair, ws_sym, current_pos["side"],
                        current_pos["size"], current_price
                    )
                    self.risk_mgr.close_position(ws_sym)
                    if target_size > 0:
                        sig = {"direction": target["side"],
                               "strength": abs(target["target_frac"]),
                               "confidence": 1.0}
                        result = self.executor.execute_signal(
                            rest_pair, ws_sym, sig, target_size, current_price
                        )
                        if result and result.get("status") in ("paper_filled", "submitted"):
                            self.risk_mgr.register_position(
                                ws_sym, target["side"], target_size, current_price
                            )
                            self.signals_generated += 1

        self.last_state = state.copy()
        self.last_action = action.copy()
        self.last_positions_snapshot = self.risk_mgr.get_positions_snapshot()

    def _rule_execution(self, signals, symbols, prices, equity):
        for ws_sym, sig in signals.items():
            if sig["direction"] == "flat":
                continue
            rest_pair = self._ws_to_rest(ws_sym)
            if not rest_pair or ws_sym not in prices:
                continue
            current_price = prices[ws_sym]
            if self.risk_mgr.check_stop_loss(ws_sym, current_price):
                pos = self.risk_mgr.get_position(ws_sym)
                if pos:
                    self._record_close_pnl(ws_sym, current_price)
                    self.executor.close_position(
                        rest_pair, ws_sym, pos["side"], pos["size"], current_price
                    )
                    self.risk_mgr.close_position(ws_sym)
                continue
            frac = min(sig["strength"] * sig["confidence"], cfg.MAX_POSITION_FRAC)
            volume = (equity * frac) / current_price if current_price > 0 else 0
            if volume <= 0:
                continue
            result = self.executor.execute_signal(
                rest_pair, ws_sym, sig, volume, current_price
            )
            if result and result.get("status") in ("paper_filled", "submitted"):
                self.risk_mgr.register_position(
                    ws_sym, sig["direction"], volume, current_price
                )
                self.signals_generated += 1

    def _build_expected_fills(self, action, symbols, prices) -> dict:
        fills = {}
        for i, sym in enumerate(symbols):
            w = float(action[i])
            if abs(w) < 0.05:
                continue
            price = prices.get(sym, 0)
            if price <= 0:
                continue
            fills[sym] = {
                "price": price,
                "volume": abs(w) * cfg.MAX_POSITION_FRAC,
                "side": "buy" if w > 0 else "sell",
            }
        return fills

    def _process_delayed_labels(self):
        now = time.time()
        label_delay = cfg.FORWARD_BARS * 60
        ready, remaining = [], []
        for feat, close_then, ts, sym in self.pending_labels:
            if now - ts >= label_delay:
                bars = self.store.get_bar_features(sym, n_bars=1)
                if bars:
                    close_now = bars[-1].get("close", 0)
                    if close_then > 0 and close_now > 0:
                        label = (close_now - close_then) / close_then
                        ready.append((feat, label))
            else:
                remaining.append((feat, close_then, ts, sym))
        self.pending_labels = remaining
        if ready:
            features = np.array([r[0] for r in ready])
            labels = np.array([r[1] for r in ready])
            self.esgd.online_update(features, labels)

    def _ws_to_rest(self, ws_sym):
        for rp, ws in cfg.PAIRS.items():
            if ws == ws_sym:
                return rp
        return None

    def _compute_pnl(self, prices: dict) -> dict:
        """Compute unrealized P&L from open positions + track realized."""
        unrealized = 0.0
        per_asset = {}

        for sym, pos in self.risk_mgr.positions.items():
            price = prices.get(sym, 0)
            if price <= 0:
                continue
            entry = pos["entry_price"]
            size = pos["size"]
            if pos["side"] == "long":
                pnl = (price - entry) * size
            else:
                pnl = (entry - price) * size
            pnl_pct = (pnl / (entry * size)) * 100 if entry * size > 0 else 0
            unrealized += pnl
            per_asset[sym] = {"pnl": pnl, "pnl_pct": pnl_pct, "side": pos["side"]}

        total = self.realized_pnl + unrealized
        return {
            "unrealized": unrealized,
            "realized": self.realized_pnl,
            "total": total,
            "per_asset": per_asset,
        }

    def _record_close_pnl(self, symbol: str, exit_price: float):
        """Record realized P&L when a position is closed."""
        pos = self.risk_mgr.get_position(symbol)
        if not pos:
            return
        entry = pos["entry_price"]
        size = pos["size"]
        if pos["side"] == "long":
            pnl = (exit_price - entry) * size
        else:
            pnl = (entry - exit_price) * size
        self.realized_pnl += pnl
        log.info(f"  💰 Closed {symbol} {pos['side']}: entry=${entry:,.2f} "
                 f"exit=${exit_price:,.2f} pnl=${pnl:,.2f} (realized total: ${self.realized_pnl:,.2f})")

    def _save_and_report(self):
        log.info("\n" + "=" * 60)
        log.info("  SESSION SUMMARY")
        log.info(f"  Bars processed:    {self.bars_processed}")
        log.info(f"  Signals generated: {self.signals_generated}")
        log.info(f"  Orders executed:   {len(self.executor.order_log)}")
        log.info(f"  Open positions:    {len(self.risk_mgr.positions)}")
        log.info(f"  Realized P&L:      ${self.realized_pnl:+.2f}")
        if self.starting_equity:
            log.info(f"  Starting equity:   ${self.starting_equity:,.2f}")
            log.info(f"  Return:            {(self.realized_pnl / self.starting_equity) * 100:+.2f}%")

        if self.use_rl and self.rl_agent:
            log.info(f"  RL total steps:    {self.rl_agent.total_steps}")
            log.info(f"  RL training steps: {self.rl_agent.training_steps}")
            log.info(f"  RL buffer size:    {len(self.rl_agent.replay_buffer)}")
            if self.outcome_tracker:
                stats = self.outcome_tracker.get_stats()
                log.info(f"  Outcome fills:     {stats['total_fills']}")
                log.info(f"  Outcome misses:    {stats['total_missed']}")
                log.info(f"  Avg slippage:      {stats['avg_slippage_pct']:.4f}%")
                log.info(f"  Fill rate:         {stats['fill_rate']:.1%}")
            rl_dir = os.path.join(self.model_dir, "rl")
            self.rl_agent.save(rl_dir)

        esgd_path = os.path.join(self.model_dir, "esgd.pkl")
        self.esgd.save(esgd_path)
        log.info("=" * 60)


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  ENTRY POINT
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def main():
    parser = argparse.ArgumentParser(description="MSIF-OEM Trading Engine")
    parser.add_argument("--live", action="store_true", help="Enable live trading")
    parser.add_argument("--no-rl", action="store_true", help="Disable RL agent")
    parser.add_argument("--model", type=str, default=cfg.MODEL_DIR, help="Model dir")
    parser.add_argument("--backfill-days", type=int, default=7,
                        help="Days of history for initial backfill (default: 7)")
    parser.add_argument("--retrain", action="store_true",
                        help="Force re-backfill and retrain even if models exist")
    args = parser.parse_args()

    # ── Auto-bootstrap: ensure models exist ──
    try:
        ensure_models_exist(
            model_dir=args.model,
            data_dir=cfg.DATA_DIR,
            backfill_days=args.backfill_days,
            force_retrain=args.retrain,
        )
    except Exception as e:
        log.error(f"Setup failed: {e}")
        sys.exit(1)

    # ── Start trading ──
    engine = TradingEngine(
        model_dir=args.model,
        paper_trading=not args.live,
        use_rl=not args.no_rl,
    )

    loop = asyncio.new_event_loop()
    def shutdown(sig, frame):
        log.info("Shutdown signal received")
        engine._stop = True
    signal.signal(signal.SIGINT, shutdown)
    signal.signal(signal.SIGTERM, shutdown)

    loop.run_until_complete(engine.start())


if __name__ == "__main__":
    main()
