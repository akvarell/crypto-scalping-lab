from __future__ import annotations

import math
import time
from typing import Iterable

import pandas as pd
import requests


BASE_URLS = [
    "https://data-api.binance.vision",
    "https://api.binance.com",
    "https://api1.binance.com",
]

STABLE_OR_FIAT_BASES = {
    "USDT", "USDC", "FDUSD", "TUSD", "DAI", "USDP", "EUR", "TRY", "GBP", "BRL",
}


def _get(path: str, params: dict | None = None, timeout: int = 15):
    last_error = None
    for base in BASE_URLS:
        try:
            response = requests.get(
                f"{base}{path}",
                params=params,
                timeout=timeout,
                headers={"User-Agent": "crypto-scalping-lab/0.9"},
            )
            response.raise_for_status()
            return response.json()
        except Exception as exc:
            last_error = exc
    raise RuntimeError(f"Binance public market data unavailable: {last_error}")


def _klines(symbol: str, interval: str = "1h", limit: int = 336) -> pd.DataFrame:
    rows = _get(
        "/api/v3/klines",
        {"symbol": symbol, "interval": interval, "limit": min(int(limit), 1000)},
    )
    if not isinstance(rows, list) or not rows:
        raise RuntimeError(f"No klines returned for {symbol}")

    df = pd.DataFrame(
        rows,
        columns=[
            "open_time", "open", "high", "low", "close", "volume",
            "close_time", "quote_volume", "trades", "taker_base",
            "taker_quote", "ignore",
        ],
    )
    df["open_time"] = pd.to_datetime(df["open_time"], unit="ms", utc=True)
    for col in ["open", "high", "low", "close", "volume", "quote_volume"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df["trades"] = pd.to_numeric(df["trades"], errors="coerce")
    return df.set_index("open_time")


def _atr_pct(df: pd.DataFrame, period: int = 14) -> float:
    prev_close = df["close"].shift(1)
    tr = pd.concat(
        [
            df["high"] - df["low"],
            (df["high"] - prev_close).abs(),
            (df["low"] - prev_close).abs(),
        ],
        axis=1,
    ).max(axis=1)
    atr = tr.rolling(period, min_periods=period).mean()
    atr_pct = atr / df["close"].replace(0.0, float("nan")) * 100.0
    return float(atr_pct.dropna().median()) if not atr_pct.dropna().empty else 0.0


def _pct_rank(series: pd.Series) -> pd.Series:
    if series.empty:
        return series
    return series.rank(pct=True, method="average") * 100.0


def _is_research_pair(symbol_info: dict) -> bool:
    symbol = str(symbol_info.get("symbol", ""))
    base = str(symbol_info.get("baseAsset", ""))
    quote = str(symbol_info.get("quoteAsset", ""))
    status = str(symbol_info.get("status", ""))

    if quote != "USDT" or status != "TRADING":
        return False
    if base in STABLE_OR_FIAT_BASES:
        return False
    if any(base.endswith(suffix) for suffix in ("UP", "DOWN", "BULL", "BEAR")):
        return False
    if not symbol.endswith("USDT"):
        return False
    return True


def build_universe_screener(
    *,
    max_symbols: int = 30,
    lookback_hours: int = 336,
    min_quote_volume_usd: float = 10_000_000.0,
) -> pd.DataFrame:
    """Build a research screener for liquid/volatile Binance USDT spot pairs.

    The score is descriptive only; BTC correlation is shown separately and is
    deliberately not used to rank symbols.
    """
    exchange_info = _get("/api/v3/exchangeInfo")
    tickers = _get("/api/v3/ticker/24hr")

    valid_symbols = {
        item["symbol"]: item
        for item in exchange_info.get("symbols", [])
        if _is_research_pair(item)
    }

    ticker_rows = []
    for item in tickers:
        symbol = item.get("symbol")
        if symbol not in valid_symbols:
            continue

        quote_volume = float(item.get("quoteVolume") or 0.0)
        if quote_volume < float(min_quote_volume_usd):
            continue

        ticker_rows.append(
            {
                "symbol": symbol,
                "base": valid_symbols[symbol].get("baseAsset", ""),
                "quote_volume_24h": quote_volume,
                "change_24h_pct": float(item.get("priceChangePercent") or 0.0),
                "trades_24h": int(item.get("count") or 0),
            }
        )

    if not ticker_rows:
        raise RuntimeError("No liquid USDT symbols matched the screener filters.")

    liquid = (
        pd.DataFrame(ticker_rows)
        .sort_values("quote_volume_24h", ascending=False)
        .head(max(5, int(max_symbols)))
        .reset_index(drop=True)
    )

    btc = _klines("BTCUSDT", "1h", int(lookback_hours))
    btc_returns = btc["close"].pct_change().rename("btc_return")

    rows = []
    for _, ticker in liquid.iterrows():
        symbol = ticker["symbol"]
        try:
            candles = _klines(symbol, "1h", int(lookback_hours))
            returns = candles["close"].pct_change().rename("asset_return")

            joined = pd.concat([returns, btc_returns], axis=1).dropna()
            corr = (
                float(joined["asset_return"].corr(joined["btc_return"]))
                if len(joined) >= 24
                else float("nan")
            )

            hourly_std = float(returns.dropna().std()) if not returns.dropna().empty else 0.0
            realized_vol_daily_pct = hourly_std * math.sqrt(24) * 100.0
            avg_abs_move_pct = (
                float(returns.dropna().abs().mean()) * 100.0
                if not returns.dropna().empty
                else 0.0
            )
            atr_pct = _atr_pct(candles)
            quote_volume_7d_avg = float(candles["quote_volume"].tail(168).sum() / 7.0)
            trades_7d_avg = float(candles["trades"].tail(168).sum() / 7.0)

            rows.append(
                {
                    "Symbol": symbol,
                    "Base": ticker["base"],
                    "24h turnover $": float(ticker["quote_volume_24h"]),
                    "7d avg turnover $": quote_volume_7d_avg,
                    "24h change %": float(ticker["change_24h_pct"]),
                    "Dailyized realized vol %": realized_vol_daily_pct,
                    "ATR %": atr_pct,
                    "Avg abs 1h move %": avg_abs_move_pct,
                    "BTC corr": corr,
                    "24h trades": int(ticker["trades_24h"]),
                    "7d avg trades/day": trades_7d_avg,
                }
            )
        except Exception:
            continue
        time.sleep(0.03)

    out = pd.DataFrame(rows)
    if out.empty:
        raise RuntimeError("Could not calculate historical metrics for the liquid symbol set.")

    out["Liquidity pct"] = _pct_rank(out["7d avg turnover $"])
    out["Volatility pct"] = _pct_rank(out["Dailyized realized vol %"])
    out["ATR pct rank"] = _pct_rank(out["ATR %"])
    out["Activity pct"] = _pct_rank(out["7d avg trades/day"])

    out["Research score"] = (
        0.30 * out["Liquidity pct"]
        + 0.35 * out["Volatility pct"]
        + 0.25 * out["ATR pct rank"]
        + 0.10 * out["Activity pct"]
    )

    return out.sort_values("Research score", ascending=False).reset_index(drop=True)
