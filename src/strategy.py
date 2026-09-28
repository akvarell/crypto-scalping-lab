from __future__ import annotations

import pandas as pd


def generate_signals(
    df: pd.DataFrame,
    rsi_long: int = 52,
    rsi_short: int = 48,
    use_trend_filter: bool = True,
    use_volume_filter: bool = False,
    min_volume_ratio: float = 1.0,
) -> pd.DataFrame:
    out = df.copy()
    fast = out["ema_fast"]
    slow = out["ema_slow"]

    cross_up = (fast > slow) & (fast.shift(1) <= slow.shift(1))
    cross_down = (fast < slow) & (fast.shift(1) >= slow.shift(1))

    long_ok = out["rsi"] >= rsi_long
    short_ok = out["rsi"] <= rsi_short

    if use_trend_filter:
        long_ok &= out["close"] > out["trend_ema"]
        short_ok &= out["close"] < out["trend_ema"]

    if use_volume_filter:
        volume_ok = out["volume_ratio"] >= float(min_volume_ratio)
        long_ok &= volume_ok
        short_ok &= volume_ok

    out["signal"] = 0
    out.loc[cross_up & long_ok, "signal"] = 1
    out.loc[cross_down & short_ok, "signal"] = -1
    return out
