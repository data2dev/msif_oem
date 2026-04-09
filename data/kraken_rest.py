"""
Kraken REST API Client
======================
Handles all REST interactions: public data + authenticated trading.
"""

import time
import hashlib
import hmac
import base64
import urllib.parse
import logging
import requests
import config as cfg

log = logging.getLogger(__name__)


def _sign(urlpath: str, data: dict) -> dict:
    """Generate auth headers for private endpoints."""
    postdata = urllib.parse.urlencode(data)
    encoded = (str(data["nonce"]) + postdata).encode()
    message = urlpath.encode() + hashlib.sha256(encoded).digest()
    mac = hmac.new(base64.b64decode(cfg.API_SECRET), message, hashlib.sha512)
    return {
        "API-Key": cfg.API_KEY,
        "API-Sign": base64.b64encode(mac.digest()).decode(),
    }


def public(endpoint: str, params: dict = None) -> dict:
    url = f"{cfg.REST_BASE}/0/public/{endpoint}"
    r = requests.get(url, params=params, timeout=10)
    r.raise_for_status()
    body = r.json()
    if body.get("error"):
        raise RuntimeError(f"Kraken: {body['error']}")
    return body["result"]


def private(endpoint: str, params: dict = None) -> dict:
    urlpath = f"/0/private/{endpoint}"
    url = f"{cfg.REST_BASE}{urlpath}"
    data = params or {}
    data["nonce"] = str(int(time.time() * 1000))
    r = requests.post(url, headers=_sign(urlpath, data), data=data, timeout=10)
    r.raise_for_status()
    body = r.json()
    if body.get("error"):
        raise RuntimeError(f"Kraken: {body['error']}")
    return body["result"]


# ── Public data helpers ──────────────────────

def get_ohlcv(pair: str, interval: int = 1, since: int = None) -> list:
    """Fetch OHLCV candles. Returns list of [ts, o, h, l, c, vwap, vol, count]."""
    params = {"pair": pair, "interval": interval}
    if since:
        params["since"] = since
    result = public("OHLC", params)
    key = [k for k in result if k != "last"][0]
    return result[key]


def get_ticker(pair: str) -> dict:
    """Fetch current ticker. Returns {ask, bid, last, vol_24h}."""
    result = public("Ticker", {"pair": pair})
    key = list(result.keys())[0]
    t = result[key]
    return {
        "ask": float(t["a"][0]),
        "bid": float(t["b"][0]),
        "last": float(t["c"][0]),
        "vol_24h": float(t["v"][1]),
    }


def get_order_book(pair: str, depth: int = 25) -> dict:
    """Fetch order book snapshot. Returns {bids: [[price, vol, ts], ...], asks: ...}."""
    result = public("Depth", {"pair": pair, "count": depth})
    key = list(result.keys())[0]
    book = result[key]
    return {
        "bids": [[float(b[0]), float(b[1]), int(b[2])] for b in book["bids"]],
        "asks": [[float(a[0]), float(a[1]), int(a[2])] for a in book["asks"]],
    }


def get_recent_trades(pair: str, since: int = None) -> list:
    """Fetch recent trades. Returns list of [price, vol, ts, side, type, misc]."""
    params = {"pair": pair}
    if since:
        params["since"] = since
    result = public("Trades", params)
    key = [k for k in result if k != "last"][0]
    return result[key]


# ── Private trading helpers ──────────────────

def get_balance() -> dict:
    """Get account balances. Returns {asset: amount_float, ...}."""
    raw = private("Balance")
    return {k: float(v) for k, v in raw.items()}


def get_trade_balance(asset: str = "ZUSD") -> dict:
    """Get trade balance summary."""
    raw = private("TradeBalance", {"asset": asset})
    return {k: float(v) for k, v in raw.items()}


def place_order(pair: str, side: str, order_type: str,
                volume: float, price: float = None, validate: bool = False) -> dict:
    """Place an order. Returns {txid: [...], descr: {...}}."""
    params = {
        "pair": pair,
        "type": side,
        "ordertype": order_type,
        "volume": str(volume),
    }
    if price is not None:
        params["price"] = str(price)
    if validate:
        params["validate"] = "true"
    return private("AddOrder", params)


def cancel_order(txid: str) -> dict:
    """Cancel an order by TXID."""
    return private("CancelOrder", {"txid": txid})


def get_open_orders() -> dict:
    """Get all open orders."""
    result = private("OpenOrders")
    return result.get("open", {})
