from __future__ import annotations

import ccxt
import pandas as pd


def fetch_ohlcv(symbol: str, timeframe: str = "5m", limit: int = 500) -> pd.DataFrame:
    """Fetch public spot OHLCV candles from Binance."""
    exchange = ccxt.binance({"enableRateLimit": True})
    rows = exchange.fetch_ohlcv(symbol, timeframe=timeframe, limit=limit)
    if not rows:
        raise RuntimeError("Exchange returned no candles.")

    df = pd.DataFrame(rows, columns=["timestamp", "open", "high", "low", "close", "volume"])
    df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms", utc=True)
    df = df.set_index("timestamp")
    numeric = ["open", "high", "low", "close", "volume"]
    df[numeric] = df[numeric].astype(float)
    return df
