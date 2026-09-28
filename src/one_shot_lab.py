from __future__ import annotations

import math
import pandas as pd

from src.dynamic_scalping import _fetch_5m
from src.execution_backtest import run_strict_backtest
from src.indicators import add_indicators


VARIANTS = [
    {
        "name": "Repeated extreme · control",
        "mode": "repeated",
        "volume_ratio": 0.0,
        "range_atr": 0.0,
    },
    {
        "name": "One-shot extreme",
        "mode": "cross",
        "volume_ratio": 0.0,
        "range_atr": 0.0,
    },
    {
        "name": "One-shot + volume spike",
        "mode": "cross",
        "volume_ratio": 1.5,
        "range_atr": 0.0,
    },
    {
        "name": "One-shot + capitulation",
        "mode": "cross",
        "volume_ratio": 1.5,
        "range_atr": 1.25,
    },
]


def _build_signal(
    market: pd.DataFrame,
    *,
    mode: str,
    min_volume_ratio: float,
    min_range_atr: float,
) -> pd.DataFrame:
    out = add_indicators(
        market,
        ema_fast=9,
        ema_slow=21,
        trend_ema=100,
        rsi_period=14,
        atr_period=14,
        volume_period=20,
    )

    mean = out["close"].rolling(20, min_periods=20).mean()
    std = out["close"].rolling(20, min_periods=20).std()
    z = (out["close"] - mean) / std.replace(0.0, float("nan"))

    if mode == "repeated":
        long_event = (z <= -2.0) & (out["rsi"] <= 30)
        short_event = (z >= 2.0) & (out["rsi"] >= 70)
    else:
        long_event = (
            (z <= -2.0)
            & (z.shift(1) > -2.0)
            & (out["rsi"] <= 30)
        )
        short_event = (
            (z >= 2.0)
            & (z.shift(1) < 2.0)
            & (out["rsi"] >= 70)
        )

    if float(min_volume_ratio) > 0:
        volume_ok = out["volume_ratio"] >= float(min_volume_ratio)
        long_event &= volume_ok
        short_event &= volume_ok

    if float(min_range_atr) > 0:
        candle_range = out["high"] - out["low"]
        shock_ok = candle_range >= out["atr"] * float(min_range_atr)
        long_event &= shock_ok
        short_event &= shock_ok

    out["signal"] = 0
    out.loc[long_event, "signal"] = 1
    out.loc[short_event, "signal"] = -1
    return out


def _safe_pf(value: float) -> float:
    return 5.0 if math.isinf(float(value)) else float(value)


def run_one_shot_lab(
    rolling_details: pd.DataFrame,
    *,
    forward_days: int,
    fee_bps: float,
    slippage_bps: float,
    max_periods: int = 12,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    if rolling_details is None or rolling_details.empty:
        raise ValueError("Run Rolling Universe Validation first.")

    selected = rolling_details[["Period", "Selection time", "Symbol"]].drop_duplicates().copy()
    selected["Selection time"] = pd.to_datetime(selected["Selection time"], utc=True)
    keep_periods = sorted(selected["Period"].unique())[-int(max_periods):]
    selected = selected[selected["Period"].isin(keep_periods)]

    cache: dict[tuple[str, pd.Timestamp], pd.DataFrame] = {}
    rows = []

    for _, sel in selected.iterrows():
        period = int(sel["Period"])
        selection_time = pd.Timestamp(sel["Selection time"])
        symbol = str(sel["Symbol"])
        key = (symbol, selection_time)

        if key not in cache:
            try:
                cache[key] = _fetch_5m(
                    symbol,
                    selection_time,
                    selection_time + pd.Timedelta(days=int(forward_days)),
                )
            except Exception:
                cache[key] = pd.DataFrame()

        market = cache[key]
        if market.empty or len(market) < 200:
            continue

        for variant in VARIANTS:
            signal_df = _build_signal(
                market,
                mode=variant["mode"],
                min_volume_ratio=float(variant["volume_ratio"]),
                min_range_atr=float(variant["range_atr"]),
            )

            gross = run_strict_backtest(
                signal_df,
                start_cash=10_000.0,
                fee_bps=0.0,
                slippage_bps=0.0,
                stop_atr=1.0,
                take_atr=1.25,
                side="BOTH",
                cooldown_bars=0,
            )
            net = run_strict_backtest(
                signal_df,
                start_cash=10_000.0,
                fee_bps=float(fee_bps),
                slippage_bps=float(slippage_bps),
                stop_atr=1.0,
                take_atr=1.25,
                side="BOTH",
                cooldown_bars=0,
            )

            gm = gross["metrics"]
            nm = net["metrics"]
            rows.append(
                {
                    "Variant": variant["name"],
                    "Period": period,
                    "Selection time": selection_time,
                    "Symbol": symbol,
                    "Signals": int((signal_df["signal"] != 0).sum()),
                    "Gross return %": float(gm["net_return_pct"]),
                    "Net return %": float(nm["net_return_pct"]),
                    "Cost drag pp": float(gm["net_return_pct"]) - float(nm["net_return_pct"]),
                    "Gross PF": float(gm["profit_factor"]),
                    "Net PF": float(nm["profit_factor"]),
                    "Trades": int(nm["trades"]),
                    "Win rate %": float(nm["win_rate_pct"]),
                    "Max DD %": float(nm["max_drawdown_pct"]),
                    "Avg trade %": float(nm["avg_trade_pct"]),
                }
            )

    details = pd.DataFrame(rows)
    if details.empty:
        raise RuntimeError("No 5m data was available for the one-shot lab.")

    period_rows = []
    for (variant, period), group in details.groupby(["Variant", "Period"], sort=True):
        period_rows.append(
            {
                "Variant": variant,
                "Period": int(period),
                "Selection time": group["Selection time"].min(),
                "Symbols": int(group["Symbol"].nunique()),
                "Gross portfolio return %": float(group["Gross return %"].mean()),
                "Net portfolio return %": float(group["Net return %"].mean()),
                "Cost drag pp": float(group["Cost drag pp"].mean()),
                "Trades": int(group["Trades"].sum()),
                "Signals": int(group["Signals"].sum()),
                "Avg trade %": float(group["Avg trade %"].mean()),
            }
        )
    periods = pd.DataFrame(period_rows)

    summary_rows = []
    for variant, group in periods.groupby("Variant", sort=False):
        net = group["Net portfolio return %"].astype(float)
        gross = group["Gross portfolio return %"].astype(float)
        detail = details[details["Variant"] == variant]
        net_pf = detail["Net PF"].map(_safe_pf)

        summary_rows.append(
            {
                "Variant": variant,
                "Periods": int(len(group)),
                "Positive net periods": f"{int((net > 0).sum())}/{len(group)}",
                "Avg gross return %": float(gross.mean()),
                "Avg net return %": float(net.mean()),
                "Median net return %": float(net.median()),
                "Compounded net %": (float((1.0 + net / 100.0).prod()) - 1.0) * 100.0,
                "Avg cost drag pp": float((gross - net).mean()),
                "Worst period %": float(net.min()),
                "Best period %": float(net.max()),
                "Total trades": int(group["Trades"].sum()),
                "Total signals": int(group["Signals"].sum()),
                "Avg trade %": float(detail["Avg trade %"].mean()),
                "Avg net PF": float(net_pf.mean()),
            }
        )

    return pd.DataFrame(summary_rows), periods, details
