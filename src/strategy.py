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
    """Original EMA-cross strategy kept as the baseline and optimizer target."""
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


def generate_family_signals(
    df: pd.DataFrame,
    family: str,
    *,
    breakout_window: int = 20,
    mean_window: int = 20,
    z_entry: float = 2.0,
) -> pd.DataFrame:
    """Reference signals for strategy-family research.

    These rules are intentionally fixed/simple. The family benchmark is meant
    to compare hypotheses before parameter optimization.
    """
    out = df.copy()
    out["signal"] = 0

    if family == "EMA Cross":
        return generate_signals(
            out,
            rsi_long=52,
            rsi_short=48,
            use_trend_filter=True,
            use_volume_filter=False,
        )

    if family == "Trend Pullback":
        trend_long = (
            (out["ema_fast"] > out["ema_slow"])
            & (out["close"] > out["trend_ema"])
            & (out["close"] > out["ema_fast"])
        )
        trend_short = (
            (out["ema_fast"] < out["ema_slow"])
            & (out["close"] < out["trend_ema"])
            & (out["close"] < out["ema_fast"])
        )
        rsi_reclaim_long = (out["rsi"] > 50) & (out["rsi"].shift(1) <= 50)
        rsi_reclaim_short = (out["rsi"] < 50) & (out["rsi"].shift(1) >= 50)

        out.loc[trend_long & rsi_reclaim_long, "signal"] = 1
        out.loc[trend_short & rsi_reclaim_short, "signal"] = -1
        return out

    if family == "Donchian Breakout":
        prior_high = out["high"].rolling(int(breakout_window), min_periods=int(breakout_window)).max().shift(1)
        prior_low = out["low"].rolling(int(breakout_window), min_periods=int(breakout_window)).min().shift(1)
        volume_ok = out["volume_ratio"] >= 1.0

        long_break = (
            (out["close"] > prior_high)
            & (out["close"] > out["trend_ema"])
            & (out["ema_fast"] > out["ema_slow"])
            & volume_ok
        )
        short_break = (
            (out["close"] < prior_low)
            & (out["close"] < out["trend_ema"])
            & (out["ema_fast"] < out["ema_slow"])
            & volume_ok
        )

        out.loc[long_break, "signal"] = 1
        out.loc[short_break, "signal"] = -1
        return out

    if family == "Mean Reversion":
        mean = out["close"].rolling(int(mean_window), min_periods=int(mean_window)).mean()
        std = out["close"].rolling(int(mean_window), min_periods=int(mean_window)).std()
        z = (out["close"] - mean) / std.replace(0.0, float("nan"))

        out.loc[(z <= -float(z_entry)) & (out["rsi"] <= 30), "signal"] = 1
        out.loc[(z >= float(z_entry)) & (out["rsi"] >= 70), "signal"] = -1
        return out

    raise ValueError(f"Unknown strategy family: {family}")



def generate_mean_reversion_variant(
    df: pd.DataFrame,
    variant: str,
    *,
    mean_window: int = 20,
    z_entry: float = 2.0,
    slope_bars: int = 12,
) -> pd.DataFrame:
    """Fixed mean-reversion research variants for regime diagnostics."""
    out = df.copy()
    out["signal"] = 0

    mean = out["close"].rolling(int(mean_window), min_periods=int(mean_window)).mean()
    std = out["close"].rolling(int(mean_window), min_periods=int(mean_window)).std()
    z = (out["close"] - mean) / std.replace(0.0, float("nan"))

    extreme_long = (z <= -float(z_entry)) & (out["rsi"] <= 30)
    extreme_short = (z >= float(z_entry)) & (out["rsi"] >= 70)

    if variant == "Extreme entry":
        out.loc[extreme_long, "signal"] = 1
        out.loc[extreme_short, "signal"] = -1
        return out

    reclaim_long = (
        (z.shift(1) <= -float(z_entry))
        & (z > -float(z_entry))
        & (out["rsi"] > out["rsi"].shift(1))
        & (out["rsi"] <= 45)
    )
    reclaim_short = (
        (z.shift(1) >= float(z_entry))
        & (z < float(z_entry))
        & (out["rsi"] < out["rsi"].shift(1))
        & (out["rsi"] >= 55)
    )

    if variant == "Reclaim entry":
        out.loc[reclaim_long, "signal"] = 1
        out.loc[reclaim_short, "signal"] = -1
        return out

    trend_move = (out["trend_ema"] - out["trend_ema"].shift(int(slope_bars))).abs()
    flat_regime = trend_move <= (out["atr"] * 1.25)

    if variant == "Reclaim + flat regime":
        out.loc[reclaim_long & flat_regime, "signal"] = 1
        out.loc[reclaim_short & flat_regime, "signal"] = -1
        return out

    if variant == "Reclaim + flat + vol guard":
        atr_pct = out["atr"] / out["close"].replace(0.0, float("nan"))
        vol_ceiling = atr_pct.rolling(200, min_periods=100).quantile(0.80)
        vol_floor = atr_pct.rolling(200, min_periods=100).quantile(0.20)
        normal_vol = (atr_pct <= vol_ceiling) & (atr_pct >= vol_floor)

        out.loc[reclaim_long & flat_regime & normal_vol, "signal"] = 1
        out.loc[reclaim_short & flat_regime & normal_vol, "signal"] = -1
        return out

    raise ValueError(f"Unknown mean-reversion variant: {variant}")
