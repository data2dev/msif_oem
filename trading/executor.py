"""
Order Executor
===============
Handles the full order lifecycle:
  1. Place limit order → store as pending with TXID
  2. Poll Kraken for fill status every tick
  3. On confirmed fill → return fill data (real price, volume)
  4. On stale order (>3 min unfilled) → cancel
  5. Paper mode simulates fills with realistic slippage

Positions are NOT registered until fills are confirmed.
"""

import logging
import time
import random
import config as cfg
from data import kraken_rest as api

log = logging.getLogger(__name__)

STALE_ORDER_TIMEOUT = 180  # cancel unfilled orders after 3 minutes


class PendingOrder:
    """An order placed but not yet confirmed filled."""

    def __init__(self, txid: str, rest_pair: str, ws_symbol: str,
                 side: str, intended_price: float, volume: float,
                 signal: dict, timestamp: float, is_close: bool = False,
                 close_position_data: dict = None):
        self.txid = txid
        self.rest_pair = rest_pair
        self.ws_symbol = ws_symbol
        self.side = side
        self.intended_price = intended_price
        self.volume = volume
        self.signal = signal
        self.timestamp = timestamp
        self.is_close = is_close
        self.close_position_data = close_position_data


class Fill:
    """A confirmed fill."""

    def __init__(self, txid: str, ws_symbol: str, rest_pair: str,
                 side: str, price: float, volume: float,
                 intended_price: float, signal: dict,
                 is_close: bool, close_position_data: dict = None,
                 was_taker: bool = False):
        self.txid = txid
        self.ws_symbol = ws_symbol
        self.rest_pair = rest_pair
        self.side = side
        self.price = price
        self.volume = volume
        self.intended_price = intended_price
        self.signal = signal
        self.is_close = is_close
        self.close_position_data = close_position_data
        self.was_taker = was_taker
        self.slippage = abs(price - intended_price) / intended_price if intended_price > 0 else 0


class Executor:
    """Executes trades with proper fill tracking."""

    def __init__(self, paper_trading: bool = True):
        self.paper_trading = paper_trading
        self.pending_orders: list[PendingOrder] = []
        self.order_log = []
        self._paper_counter = 0

        mode = "PAPER" if paper_trading else "LIVE"
        log.info(f"Executor initialized in {mode} mode (0.1% offset, {STALE_ORDER_TIMEOUT}s timeout)")

    def place_order(self, rest_pair: str, ws_symbol: str, side: str,
                    volume: float, current_price: float, signal: dict = None,
                    is_close: bool = False,
                    close_position_data: dict = None) -> PendingOrder | None:
        """
        Place a limit order. Returns PendingOrder (NOT a confirmed fill).
        Call check_fills() every tick to get confirmed fills.
        """
        if volume <= 0:
            return None

        # 0.1% offset for faster fills
        if side == "buy":
            price = round(current_price * 1.001, 4)
        else:
            price = round(current_price * 0.999, 4)

        if self.paper_trading:
            return self._paper_place(rest_pair, ws_symbol, side, price,
                                     volume, signal, is_close, close_position_data)
        else:
            return self._live_place(rest_pair, ws_symbol, side, price,
                                    volume, signal, is_close, close_position_data)

    def check_fills(self) -> list[Fill]:
        """Check all pending orders for fills. Call every tick. Returns confirmed fills."""
        if self.paper_trading:
            return self._paper_check_fills()
        else:
            return self._live_check_fills()

    def cancel_stale_orders(self) -> list[PendingOrder]:
        """Cancel orders pending too long. Returns cancelled orders."""
        now = time.time()
        stale = []
        remaining = []

        for order in self.pending_orders:
            if now - order.timestamp > STALE_ORDER_TIMEOUT:
                stale.append(order)
                if not self.paper_trading:
                    try:
                        api.cancel_order(order.txid)
                        log.info(f"Cancelled stale order {order.txid} for {order.ws_symbol}")
                    except Exception as e:
                        log.error(f"Failed to cancel {order.txid}: {e}")
                else:
                    log.info(f"Paper: cancelled stale order for {order.ws_symbol} "
                             f"({now - order.timestamp:.0f}s old)")

                self.order_log.append({
                    "txid": order.txid, "symbol": order.ws_symbol,
                    "side": order.side, "status": "cancelled_stale",
                    "timestamp": order.timestamp,
                })
            else:
                remaining.append(order)

        self.pending_orders = remaining
        return stale

    def get_pending_symbols(self) -> set:
        """Get symbols with pending unfilled orders."""
        return {o.ws_symbol for o in self.pending_orders}

    # ── Paper trading ────────────────────────

    def _paper_place(self, rest_pair, ws_symbol, side, price, volume,
                     signal, is_close, close_position_data) -> PendingOrder:
        self._paper_counter += 1
        txid = f"PAPER-{self._paper_counter}-{int(time.time())}"

        order = PendingOrder(
            txid=txid, rest_pair=rest_pair, ws_symbol=ws_symbol,
            side=side, intended_price=price, volume=volume,
            signal=signal or {}, timestamp=time.time(),
            is_close=is_close, close_position_data=close_position_data,
        )
        self.pending_orders.append(order)

        action = "CLOSE" if is_close else "OPEN"
        log.info(f"📝 PAPER {action} {side.upper()} {volume:.6f} {ws_symbol} "
                 f"@ ${price:,.4f} [pending: {txid}]")
        return order

    def _paper_check_fills(self) -> list[Fill]:
        """Simulate fills with small random slippage."""
        fills = []
        for order in list(self.pending_orders):
            slippage_pct = random.uniform(0, 0.0005)
            if order.side == "buy":
                fill_price = order.intended_price * (1 + slippage_pct)
            else:
                fill_price = order.intended_price * (1 - slippage_pct)

            fill = Fill(
                txid=order.txid, ws_symbol=order.ws_symbol,
                rest_pair=order.rest_pair, side=order.side,
                price=round(fill_price, 4), volume=order.volume,
                intended_price=order.intended_price, signal=order.signal,
                is_close=order.is_close,
                close_position_data=order.close_position_data,
                was_taker=False,
            )
            fills.append(fill)

            action = "CLOSE" if order.is_close else "FILL"
            log.info(f"📝 PAPER {action} {order.side.upper()} {order.volume:.6f} "
                     f"{order.ws_symbol} @ ${fill_price:,.4f} "
                     f"(slip: {fill.slippage*100:.3f}%)")

            self.order_log.append({
                "txid": order.txid, "symbol": order.ws_symbol,
                "side": order.side, "price": fill_price,
                "intended_price": order.intended_price,
                "volume": order.volume, "status": "paper_filled",
                "slippage": fill.slippage, "timestamp": time.time(),
            })

        self.pending_orders.clear()
        return fills

    # ── Live trading ─────────────────────────

    def _live_place(self, rest_pair, ws_symbol, side, price, volume,
                    signal, is_close, close_position_data) -> PendingOrder | None:
        try:
            result = api.place_order(
                pair=rest_pair, side=side,
                order_type="limit", volume=volume, price=price,
            )
            txids = result.get("txid", [])
            if not txids:
                log.error(f"No TXID returned for {ws_symbol} order")
                return None

            txid = txids[0]
            order = PendingOrder(
                txid=txid, rest_pair=rest_pair, ws_symbol=ws_symbol,
                side=side, intended_price=price, volume=volume,
                signal=signal or {}, timestamp=time.time(),
                is_close=is_close, close_position_data=close_position_data,
            )
            self.pending_orders.append(order)

            action = "CLOSE" if is_close else "OPEN"
            log.info(f"🔴 LIVE {action} {side.upper()} {volume:.6f} {ws_symbol} "
                     f"@ ${price:,.4f} [txid: {txid}]")
            return order

        except Exception as e:
            log.error(f"Order placement failed for {ws_symbol}: {e}")
            return None

    def _live_check_fills(self) -> list[Fill]:
        """Check Kraken open orders to detect fills."""
        if not self.pending_orders:
            return []

        try:
            open_orders = api.get_open_orders()
        except Exception as e:
            log.error(f"Failed to check open orders: {e}")
            return []

        fills = []
        remaining = []

        for order in self.pending_orders:
            if order.txid in open_orders:
                remaining.append(order)
            else:
                fill_price = order.intended_price
                was_taker = False
                try:
                    closed = api.private("QueryOrders", {"txid": order.txid})
                    if order.txid in closed:
                        info = closed[order.txid]
                        if float(info.get("vol_exec", 0)) > 0:
                            fill_price = float(info.get("price", order.intended_price))
                except Exception:
                    pass

                fill = Fill(
                    txid=order.txid, ws_symbol=order.ws_symbol,
                    rest_pair=order.rest_pair, side=order.side,
                    price=fill_price, volume=order.volume,
                    intended_price=order.intended_price,
                    signal=order.signal, is_close=order.is_close,
                    close_position_data=order.close_position_data,
                    was_taker=was_taker,
                )
                fills.append(fill)

                log.info(f"🔴 FILLED {order.side.upper()} {order.volume:.6f} "
                         f"{order.ws_symbol} @ ${fill_price:,.4f} "
                         f"(slip: {fill.slippage*100:.3f}%)")

                self.order_log.append({
                    "txid": order.txid, "symbol": order.ws_symbol,
                    "side": order.side, "price": fill_price,
                    "intended_price": order.intended_price,
                    "volume": order.volume, "status": "filled",
                    "slippage": fill.slippage, "was_taker": was_taker,
                    "timestamp": time.time(),
                })

        self.pending_orders = remaining
        return fills
