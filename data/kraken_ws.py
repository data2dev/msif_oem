"""
Kraken WebSocket Client
=======================
Maintains persistent connections for order book (L2) and trade streams.
Feeds raw data into the DataStore for feature computation.
Auto-reconnects on disconnect.
"""

import asyncio
import json
import logging
import time
import websockets
import config as cfg

log = logging.getLogger(__name__)


class KrakenWebSocket:
    """Async WebSocket client for Kraken v2 public feeds."""

    def __init__(self, store):
        """
        Args:
            store: DataStore instance to push updates into.
        """
        self.store = store
        self.ws = None
        self._running = False
        self._reconnect_delay = 1

    async def start(self):
        """Connect and begin streaming. Reconnects on failure."""
        self._running = True
        while self._running:
            try:
                log.info(f"Connecting to {cfg.WS_PUBLIC}")
                async with websockets.connect(
                    cfg.WS_PUBLIC,
                    ping_interval=30,
                    ping_timeout=10,
                ) as ws:
                    self.ws = ws
                    self._reconnect_delay = 1
                    log.info("WebSocket connected")

                    await self._subscribe(ws)
                    await self._listen(ws)

            except (websockets.ConnectionClosed, ConnectionError, OSError) as e:
                log.warning(f"WS disconnected: {e}. Reconnecting in {self._reconnect_delay}s")
                await asyncio.sleep(self._reconnect_delay)
                self._reconnect_delay = min(self._reconnect_delay * 2, 30)
            except Exception as e:
                log.error(f"WS unexpected error: {e}", exc_info=True)
                await asyncio.sleep(5)

    async def stop(self):
        self._running = False
        if self.ws:
            await self.ws.close()

    async def _subscribe(self, ws):
        """Subscribe to book and trade channels for all pairs."""
        # Order book L2
        await ws.send(json.dumps({
            "method": "subscribe",
            "params": {
                "channel": "book",
                "symbol": cfg.PAIR_LIST_WS,
                "depth": cfg.BOOK_DEPTH,
            },
        }))
        log.info(f"Subscribed to book: {cfg.PAIR_LIST_WS}")

        # Trades
        await ws.send(json.dumps({
            "method": "subscribe",
            "params": {
                "channel": "trade",
                "symbol": cfg.PAIR_LIST_WS,
            },
        }))
        log.info(f"Subscribed to trade: {cfg.PAIR_LIST_WS}")

    async def _listen(self, ws):
        """Main message loop."""
        async for raw in ws:
            try:
                msg = json.loads(raw)
                channel = msg.get("channel")

                if channel == "book":
                    self._handle_book(msg)
                elif channel == "trade":
                    self._handle_trade(msg)
                elif channel == "heartbeat":
                    pass
                elif channel == "status":
                    status = msg.get("data", [{}])[0].get("system", "?")
                    log.info(f"Kraken system status: {status}")

            except Exception as e:
                log.error(f"Message processing error: {e}", exc_info=True)

    def _handle_book(self, msg):
        """Process book snapshot or update into the store."""
        msg_type = msg.get("type")
        for entry in msg.get("data", []):
            symbol = entry.get("symbol", "")
            bids = [(float(b["price"]), float(b["qty"])) for b in entry.get("bids", [])]
            asks = [(float(a["price"]), float(a["qty"])) for a in entry.get("asks", [])]

            if msg_type == "snapshot":
                self.store.book_snapshot(symbol, bids, asks)
            elif msg_type == "update":
                self.store.book_update(symbol, bids, asks)

    def _handle_trade(self, msg):
        """Process trade ticks into the store."""
        for trade in msg.get("data", []):
            symbol = trade.get("symbol", "")
            self.store.trade_tick(
                symbol=symbol,
                price=float(trade.get("price", 0)),
                qty=float(trade.get("qty", 0)),
                side=trade.get("side", ""),
                timestamp=trade.get("timestamp", ""),
            )
