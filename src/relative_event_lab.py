from __future__ import annotations

import math
import pandas as pd

from src.dynamic_scalping import _fetch_5m
from src.execution_backtest import run_strict_backtest
from src.indicators import add_indicators


VARIANTS = [
    {
        "name": "Relative momentum shock",
        "stop_atr": 1.25,
        "take_atr": 2.00,
    },
    {
        "name": "Decorrelation breakout",
        "stop_atr": 1.25,
        "take_atr": 2.00,
    },
    {
        "name": "Capitulation reversal",
        "stop_atr": 1.00,
        "take_atr": 1.50,
    },
]


def _prepare_relative_features(
    coin: pd.DataFrame,
    btc: pd.DataFrame,
) -> pd.DataFrame:
    out = add_indicators(
        coin,
        ema_fast=9,
        ema_slow=21,
        trend_ema=100,
        rsi_period=14,
        atr_period=14,
        volume_period=20,
    )

    aligned = pd.DataFrame(index=out.index)
    aligned["coin_close"] = out["close"]
    aligned["btc_close"] = btc["close"].reindex(out.index).ffill()

    coin_ret = aligned["coin_close"].pct_change()
    btc_ret = aligned["btc_close"].pct_change()

    cov = coin_ret.rolling(96, min_periods=48).cov(btc_ret)
    var = btc_ret.rolling(96, min_periods=48).var().replace(0.0, float("nan"))
    beta = (cov / var).clip(-3.0, 3.0)

    coin_ret_3 = aligned["coin_close"].pct_change(3)
    btc_ret_3 = aligned["btc_close"].pct_change(3)
    out["btc_corr"] = coin_ret.rolling(96, min_periods=48).corr(btc_ret)
    out["beta_btc"] = beta
    out["rel_ret_3"] = coin_ret_3 - beta * btc_ret_3
    out["atr_pct"] = out["atr"] / out["close"].replace(0.0, float("nan"))
    out["range_atr"] = (out["high"] - out["low"]) / out["atr"].replace(0.0, float("nan"))

    mean = out["close"].rolling(20, min_periods=20).mean()
    std = out["close"].rolling(20, min_periods=20).std()
    out["z20"] = (out["close"] - mean) / std.replace(0.0, float("nan"))

    out["prior_high_20"] = out["high"].rolling(20, min_periods=20).max().shift(1)
    out["prior_low_20"] = out["low"].rolling(20, min_periods=20).min().shift(1)
    return out


def _build_signal(
    coin: pd.DataFrame,
    btc: pd.DataFrame,
    variant: str,
) -> pd.DataFrame:
    out = _prepare_relative_features(coin, btc)
    out["signal"] = 0

    if variant == "Relative momentum shock":
        threshold = out["atr_pct"] * 0.80
        long_event = (
            (out["rel_ret_3"] >= threshold)
            & (out["rel_ret_3"].shift(1) < threshold.shift(1))
            & (out["volume_ratio"] >= 2.0)
            & (out["range_atr"] >= 1.0)
            & (out["ema_fast"] > out["ema_slow"])
        )
        short_event = (
            (out["rel_ret_3"] <= -threshold)
            & (out["rel_ret_3"].shift(1) > -threshold.shift(1))
            & (out["volume_ratio"] >= 2.0)
            & (out["range_atr"] >= 1.0)
            & (out["ema_fast"] < out["ema_slow"])
        )
        out.loc[long_event, "signal"] = 1
        out.loc[short_event, "signal"] = -1
        return out

    if variant == "Decorrelation breakout":
        low_corr = out["btc_corr"].abs() <= 0.45
        long_event = (
            low_corr
            & (out["close"] > out["prior_high_20"])
            & (out["close"].shift(1) <= out["prior_high_20"].shift(1))
            & (out["volume_ratio"] >= 2.0)
            & (out["range_atr"] >= 1.0)
        )
        short_event = (
            low_corr
            & (out["close"] < out["prior_low_20"])
            & (out["close"].shift(1) >= out["prior_low_20"].shift(1))
            & (out["volume_ratio"] >= 2.0)
            & (out["range_atr"] >= 1.0)
        )
        out.loc[long_event, "signal"] = 1
        out.loc[short_event, "signal"] = -1
        return out

    if variant == "Capitulation reversal":
        long_event = (
            (out["z20"] <= -2.5)
            & (out["z20"].shift(1) > -2.5)
            & (out["rsi"] <= 25)
            & (out["volume_ratio"] >= 2.0)
            & (out["range_atr"] >= 1.5)
        )
        short_event = (
            (out["z20"] >= 2.5)
            & (out["z20"].shift(1) < 2.5)
            & (out["rsi"] >= 75)
            & (out["volume_ratio"] >= 2.0)
            & (out["range_atr"] >= 1.5)
        )
        out.loc[long_event, "signal"] = 1
        out.loc[short_event, "signal"] = -1
        return out

    raise ValueError(f"Unknown event variant: {variant}")


def _safe_pf(value: float) -> float:
    return 5.0 if math.isinf(float(value)) else float(value)


def run_relative_event_lab(
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

    market_cache: dict[tuple[str, pd.Timestamp], pd.DataFrame] = {}
    btc_cache: dict[pd.Timestamp, pd.DataFrame] = {}
    rows = []

    for _, sel in selected.iterrows():
        period = int(sel["Period"])
        selection_time = pd.Timestamp(sel["Selection time"])
        symbol = str(sel["Symbol"])
        forward_end = selection_time + pd.Timedelta(days=int(forward_days))
        key = (symbol, selection_time)

        if selection_time not in btc_cache:
            try:
                btc_cache[selection_time] = _fetch_5m("BTCUSDT", selection_time, forward_end)
            except Exception:
                btc_cache[selection_time] = pd.DataFrame()

        if key not in market_cache:
            try:
                market_cache[key] = _fetch_5m(symbol, selection_time, forward_end)
            except Exception:
                market_cache[key] = pd.DataFrame()

        market = market_cache[key]
        btc = btc_cache[selection_time]
        if market.empty or btc.empty or len(market) < 200 or len(btc) < 200:
            continue

        for variant in VARIANTS:
            signal_df = _build_signal(market, btc, variant["name"])

            gross = run_strict_backtest(
                signal_df,
                start_cash=10_000.0,
                fee_bps=0.0,
                slippage_bps=0.0,
                stop_atr=float(variant["stop_atr"]),
                take_atr=float(variant["take_atr"]),
                side="BOTH",
                cooldown_bars=0,
            )
            net = run_strict_backtest(
                signal_df,
                start_cash=10_000.0,
                fee_bps=float(fee_bps),
                slippage_bps=float(slippage_bps),
                stop_atr=float(variant["stop_atr"]),
                take_atr=float(variant["take_atr"]),
                side="BOTH",
                cooldown_bars=0,
            )

            gm = gross["metrics"]
            nm = net["metrics"]
            gross_avg_trade_bps = float(gm["avg_trade_pct"]) * 100.0
            net_avg_trade_bps = float(nm["avg_trade_pct"]) * 100.0

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
                    "Gross avg trade bps": gross_avg_trade_bps,
                    "Net avg trade bps": net_avg_trade_bps,
                    "Win rate %": float(nm["win_rate_pct"]),
                    "Max DD %": float(nm["max_drawdown_pct"]),
                }
            )

    details = pd.DataFrame(rows)
    if details.empty:
        raise RuntimeError("No 5m data was available for the relative-event lab.")

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
                "Gross avg trade bps": float(group["Gross avg trade bps"].mean()),
                "Net avg trade bps": float(group["Net avg trade bps"].mean()),
            }
        )
    periods = pd.DataFrame(period_rows)

    summary_rows = []
    round_trip_cost_bps = 2.0 * (float(fee_bps) + float(slippage_bps))

    for variant, group in periods.groupby("Variant", sort=False):
        net = group["Net portfolio return %"].astype(float)
        gross = group["Gross portfolio return %"].astype(float)
        detail = details[details["Variant"] == variant]
        pf = detail["Net PF"].map(_safe_pf)
        gross_trade_bps = float(detail["Gross avg trade bps"].mean())
        net_trade_bps = float(detail["Net avg trade bps"].mean())

        summary_rows.append(
            {
                "Variant": variant,
                "Periods": int(len(group)),
                "Positive net periods": f"{int((net > 0).sum())}/{len(group)}",
                "Avg gross return %": float(gross.mean()),
                "Avg net return %": float(net.mean()),
                "Median net return %": float(net.median()),
                "Compounded net %": (float((1.0 + net / 100.0).prod()) - 1.0) * 100.0,
                "Total trades": int(group["Trades"].sum()),
                "Gross avg trade bps": gross_trade_bps,
                "Net avg trade bps": net_trade_bps,
                "Round-trip cost bps": round_trip_cost_bps,
                "Gross edge / cost": (
                    gross_trade_bps / round_trip_cost_bps
                    if round_trip_cost_bps > 0
                    else float("inf")
                ),
                "Avg net PF": float(pf.mean()),
                "Worst period %": float(net.min()),
                "Best period %": float(net.max()),
            }
        )

    return pd.DataFrame(summary_rows), periods, details
