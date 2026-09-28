from __future__ import annotations

import pandas as pd
import requests

PAIR_MAP = {
    "BTC/USDT": "XBTUSDT",
    "ETH/USDT": "ETHUSDT",
    "SOL/USDT": "SOLUSDT",
    "XRP/USDT": "XRPUSDT",
}

INTERVAL_MAP = {
    "1m": 1,
    "5m": 5,
    "15m": 15,
    "30m": 30,
    "1h": 60,
}


def fetch_ohlcv(symbol: str, timeframe: str = "5m", limit: int = 500) -> pd.DataFrame:
    """Fetch public OHLCV candles from Kraken; no API key required."""
    pair = PAIR_MAP.get(symbol)
    interval = INTERVAL_MAP.get(timeframe)
    if pair is None:
        raise ValueError(f"Unsupported symbol: {symbol}")
    if interval is None:
        raise ValueError(f"Unsupported timeframe: {timeframe}")

    response = requests.get(
        "https://api.kraken.com/0/public/OHLC",
        params={"pair": pair, "interval": interval},
        timeout=20,
        headers={"User-Agent": "crypto-scalping-lab/0.1"},
    )
    response.raise_for_status()
    payload = response.json()

    errors = payload.get("error") or []
    if errors:
        raise RuntimeError("; ".join(errors))

    result = payload.get("result") or {}
    keys = [key for key in result.keys() if key != "last"]
    if not keys:
        raise RuntimeError("Market data provider returned no candles.")

    rows = result[keys[0]]
    df = pd.DataFrame(
        rows,
        columns=["timestamp", "open", "high", "low", "close", "vwap", "volume", "count"],
    )
    df["timestamp"] = pd.to_datetime(df["timestamp"], unit="s", utc=True)
    numeric = ["open", "high", "low", "close", "vwap", "volume"]
    df[numeric] = df[numeric].astype(float)
    df = df.set_index("timestamp")
    return df[["open", "high", "low", "close", "volume"]].tail(int(limit))
