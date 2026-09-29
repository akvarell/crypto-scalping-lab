from __future__ import annotations

import pandas as pd

from src.indicators import _atr, _rsi


def add_indicators_v3(
    df: pd.DataFrame,
    *,
    ema_fast: int = 9,
    ema_slow: int = 21,
    trend_ema: int = 100,
    rsi_period: int = 14,
    atr_period: int = 14,
    volume_period: int = 20,
) -> pd.DataFrame:
    """Causal indicators for v3 research.

    The key difference from legacy research is that relative volume compares
    the current completed candle only with PRIOR candles. The current candle
    is never part of its own baseline.
    """
    out = df.copy()
    out["ema_fast"] = out["close"].ewm(span=ema_fast, adjust=False).mean()
    out["ema_slow"] = out["close"].ewm(span=ema_slow, adjust=False).mean()
    out["trend_ema"] = out["close"].ewm(span=trend_ema, adjust=False).mean()
    out["rsi"] = _rsi(out["close"], rsi_period)
    out["atr"] = _atr(out, atr_period)

    prior_volume_mean = (
        out["volume"]
        .shift(1)
        .rolling(volume_period, min_periods=volume_period)
        .mean()
    )
    out["volume_sma_prior"] = prior_volume_mean
    out["volume_ratio"] = (
        out["volume"] / prior_volume_mean.replace(0.0, float("nan"))
    )

    prior_trades_mean = (
        out["trades"]
        .shift(1)
        .rolling(volume_period, min_periods=volume_period)
        .mean()
        if "trades" in out.columns
        else pd.Series(index=out.index, dtype=float)
    )
    out["trades_ratio"] = (
        out["trades"] / prior_trades_mean.replace(0.0, float("nan"))
        if "trades" in out.columns
        else float("nan")
    )

    if "taker_base" in out.columns:
        denom = out["volume"].replace(0.0, float("nan"))
        out["taker_buy_ratio"] = out["taker_base"] / denom
        out["taker_buy_ratio_prior20"] = (
            out["taker_buy_ratio"]
            .shift(1)
            .rolling(volume_period, min_periods=volume_period)
            .mean()
        )
        out["taker_buy_ratio_delta"] = (
            out["taker_buy_ratio"] - out["taker_buy_ratio_prior20"]
        )

    return out
