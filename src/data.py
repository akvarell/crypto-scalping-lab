from __future__ import annotations

import time
from datetime import datetime, timezone

import pandas as pd
import requests

PRODUCT_MAP = {
    "BTC/USD": "BTC-USD",
    "ETH/USD": "ETH-USD",
    "SOL/USD": "SOL-USD",
    "XRP/USD": "XRP-USD",
}

GRANULARITY = {
    "1m": 60,
    "5m": 300,
    "15m": 900,
    "1h": 3600,
}


def fetch_ohlcv(symbol: str, timeframe: str = "5m", limit: int = 3000) -> pd.DataFrame:
    """Fetch historical public OHLCV candles from Coinbase Exchange.

    Coinbase returns at most about 300 candles per request, so the function
    walks backwards in time and joins multiple public requests. No API key is used.
    """
    product = PRODUCT_MAP.get(symbol)
    granularity = GRANULARITY.get(timeframe)

    if product is None:
        raise ValueError(f"Unsupported symbol: {symbol}")
    if granularity is None:
        raise ValueError(f"Unsupported timeframe: {timeframe}")

    limit = max(300, min(int(limit), 5000))
    session = requests.Session()
    session.headers.update({"User-Agent": "crypto-scalping-lab/0.4"})

    end_ts = int(datetime.now(timezone.utc).timestamp())
    end_ts -= end_ts % granularity

    rows: list[list[float]] = []
    seen: set[int] = set()
    max_batches = (limit + 299) // 300 + 2

    for _ in range(max_batches):
        if len(seen) >= limit:
            break

        batch_size = min(300, limit - len(seen))
        start_ts = end_ts - granularity * batch_size

        response = session.get(
            f"https://api.exchange.coinbase.com/products/{product}/candles",
            params={
                "granularity": granularity,
                "start": datetime.fromtimestamp(start_ts, tz=timezone.utc).isoformat(),
                "end": datetime.fromtimestamp(end_ts, tz=timezone.utc).isoformat(),
            },
            timeout=20,
        )
        response.raise_for_status()
        batch = response.json()

        if not isinstance(batch, list):
            raise RuntimeError(f"Unexpected market-data response: {batch}")
        if not batch:
            break

        oldest = None
        for item in batch:
            if not isinstance(item, list) or len(item) < 6:
                continue
            ts = int(item[0])
            oldest = ts if oldest is None else min(oldest, ts)
            if ts not in seen:
                seen.add(ts)
                rows.append(item[:6])

        if oldest is None:
            break

        end_ts = oldest - granularity
        time.sleep(0.08)

    if not rows:
        raise RuntimeError("Market data provider returned no candles.")

    # Coinbase candle order: time, low, high, open, close, volume.
    df = pd.DataFrame(
        rows,
        columns=["timestamp", "low", "high", "open", "close", "volume"],
    )
    df["timestamp"] = pd.to_datetime(df["timestamp"], unit="s", utc=True)
    numeric = ["open", "high", "low", "close", "volume"]
    df[numeric] = df[numeric].astype(float)
    df = (
        df.drop_duplicates(subset=["timestamp"])
        .sort_values("timestamp")
        .set_index("timestamp")
    )
    return df[["open", "high", "low", "close", "volume"]].tail(limit)
