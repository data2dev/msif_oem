"""
Order Executor
===============
Translates trading signals into Kraken orders.
Supports paper trading (log-only) and live modes.
"""

import logging
import time
import config as cfg
from data import kraken_rest as api

log = logging.getLogger(__name__)


class Executor:
    """Executes trades on Kraken or logs them in paper mode."""

    def __init__(self, paper_trading: bool = True):
        self.paper_trading = paper_trading
        self.order_log = []

        mode = "PAPER" if paper_trading else "LIVE"
        log.info(f"Executor initialized in {mode} trading mode")

    def execute_signal(self, rest_pair: str, ws_symbol: str, signal: dict,
                       volume: float, current_price: float) -> dict | None:
        """
        Execute a trading signal.
        
        Args:
            rest_pair: Kraken REST pair name (e.g., "XXBTZUSD")
            ws_symbol: WS symbol (e.g., "BTC/USD")
            signal: signal dict from SignalGenerator
            volume: position size in base units
            current_price: current market price
            
        Returns:
            Order result dict, or None if no action taken.
        """
        if signal["direction"] == "flat" or volume <= 0:
            return None

        side = "buy" if signal["direction"] == "long" else "sell"

        # Limit order: offset from market to avoid taker fees
        if side == "buy":
            price = round(current_price * 0.9999, 2)  # slightly below
        else:
            price = round(current_price * 1.0001, 2)  # slightly above

        order = {
            "pair": rest_pair,
            "symbol": ws_symbol,
            "side": side,
            "type": cfg.ORDER_TYPE,
            "price": price,
            "volume": round(volume, 8),
            "signal_strength": signal["strength"],
            "confidence": signal["confidence"],
            "timestamp": time.time(),
        }

        if self.paper_trading:
            return self._paper_execute(order)
        else:
            return self._live_execute(order)

    def close_position(self, rest_pair: str, ws_symbol: str, side: str,
                       volume: float, current_price: float) -> dict | None:
        """Close an existing position."""
        close_side = "sell" if side == "long" else "buy"
        price = current_price  # market-close

        order = {
            "pair": rest_pair,
            "symbol": ws_symbol,
            "side": close_side,
            "type": "market",
            "price": price,
            "volume": round(volume, 8),
            "reason": "close_position",
            "timestamp": time.time(),
        }

        if self.paper_trading:
            return self._paper_execute(order)
        else:
            return self._live_execute(order)

    def _paper_execute(self, order: dict) -> dict:
        """Log order without sending to Kraken."""
        order["status"] = "paper_filled"
        order["txid"] = f"PAPER-{int(order['timestamp'] * 1000)}"
        self.order_log.append(order)

        log.info(
            f"📝 PAPER {order['side'].upper()} {order['volume']} {order['symbol']} "
            f"@ ${order['price']:,.2f} | strength={order.get('signal_strength', 0):.4f}"
        )
        return order

    def _live_execute(self, order: dict) -> dict | None:
        """Send order to Kraken."""
        try:
            result = api.place_order(
                pair=order["pair"],
                side=order["side"],
                order_type=order["type"],
                volume=order["volume"],
                price=order["price"] if order["type"] == "limit" else None,
            )
            txids = result.get("txid", [])
            order["status"] = "submitted"
            order["txid"] = txids[0] if txids else None
            self.order_log.append(order)

            log.info(
                f"🔴 LIVE {order['side'].upper()} {order['volume']} {order['symbol']} "
                f"@ ${order['price']:,.2f} | txid={order['txid']}"
            )
            return order

        except Exception as e:
            log.error(f"Order execution failed: {e}", exc_info=True)
            order["status"] = "failed"
            order["error"] = str(e)
            self.order_log.append(order)
            return None

    def get_order_log(self) -> list:
        return self.order_log
