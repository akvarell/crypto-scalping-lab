from __future__ import annotations

import pandas as pd


def generate_signals(df: pd.DataFrame, rsi_long: int = 52, rsi_short: int = 48) -> pd.DataFrame:
    out = df.copy()
    fast = out["ema_fast"]
    slow = out["ema_slow"]

    cross_up = (fast > slow) & (fast.shift(1) <= slow.shift(1))
    cross_down = (fast < slow) & (fast.shift(1) >= slow.shift(1))

    out["signal"] = 0
    out.loc[cross_up & (out["rsi"] >= rsi_long), "signal"] = 1
    out.loc[cross_down & (out["rsi"] <= rsi_short), "signal"] = -1
    return out
