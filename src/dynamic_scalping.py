from __future__ import annotations

import math
import time

import pandas as pd
import requests

from src.backtest import run_backtest
from src.indicators import add_indicators
from src.strategy import generate_mean_reversion_variant


BASE_URLS = [
    "https://data-api.binance.vision",
    "https://api.binance.com",
    "https://api1.binance.com",
]


def _get(path: str, params: dict, timeout: int = 20):
    last_error = None
    for base in BASE_URLS:
        try:
            response = requests.get(
                f"{base}{path}",
                params=params,
                timeout=timeout,
                headers={"User-Agent": "crypto-scalping-lab/1.2"},
            )
            response.raise_for_status()
            return response.json()
        except Exception as exc:
            last_error = exc
    raise RuntimeError(f"Binance 5m data unavailable: {last_error}")


def _fetch_5m(symbol: str, start: pd.Timestamp, end: pd.Timestamp) -> pd.DataFrame:
    rows: list[list] = []
    cursor = int(start.timestamp() * 1000)
    end_ms = int(end.timestamp() * 1000)
    step_ms = 300_000

    while cursor < end_ms:
        batch = _get(
            "/api/v3/klines",
            {
                "symbol": symbol,
                "interval": "5m",
                "startTime": cursor,
                "endTime": end_ms,
                "limit": 1000,
            },
        )
        if not isinstance(batch, list) or not batch:
            break

        rows.extend(batch)
        last_open = int(batch[-1][0])
        next_cursor = last_open + step_ms
        if next_cursor <= cursor:
            break
        cursor = next_cursor

        if len(batch) < 1000:
            break
        time.sleep(0.02)

    if not rows:
        return pd.DataFrame()

    df = pd.DataFrame(
        rows,
        columns=[
            "open_time", "open", "high", "low", "close", "volume",
            "close_time", "quote_volume", "trades", "taker_base",
            "taker_quote", "ignore",
        ],
    )
    df["open_time"] = pd.to_datetime(df["open_time"], unit="ms", utc=True)
    for col in ["open", "high", "low", "close", "volume", "quote_volume", "trades", "taker_base", "taker_quote"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    return (
        df.drop_duplicates(subset=["open_time"])
        .sort_values("open_time")
        .set_index("open_time")
    )


def _vol_volume_breakout(df: pd.DataFrame) -> pd.DataFrame:
    out = add_indicators(
        df,
        ema_fast=9,
        ema_slow=21,
        trend_ema=100,
        rsi_period=14,
        atr_period=14,
        volume_period=20,
    )
    prior_high = out["high"].rolling(20, min_periods=20).max().shift(1)
    prior_low = out["low"].rolling(20, min_periods=20).min().shift(1)

    atr_pct = out["atr"] / out["close"].replace(0.0, float("nan"))
    atr_floor = atr_pct.rolling(96, min_periods=48).median()
    active_vol = atr_pct >= atr_floor

    long_condition = (
        (out["close"] > prior_high)
        & (out["close"].shift(1) <= prior_high.shift(1))
        & (out["ema_fast"] > out["ema_slow"])
        & (out["close"] > out["trend_ema"])
        & (out["rsi"] >= 55)
        & (out["volume_ratio"] >= 1.5)
        & active_vol
    )
    short_condition = (
        (out["close"] < prior_low)
        & (out["close"].shift(1) >= prior_low.shift(1))
        & (out["ema_fast"] < out["ema_slow"])
        & (out["close"] < out["trend_ema"])
        & (out["rsi"] <= 45)
        & (out["volume_ratio"] >= 1.5)
        & active_vol
    )

    out["signal"] = 0
    out.loc[long_condition, "signal"] = 1
    out.loc[short_condition, "signal"] = -1
    return out


def _momentum_pullback(df: pd.DataFrame) -> pd.DataFrame:
    out = add_indicators(
        df,
        ema_fast=9,
        ema_slow=21,
        trend_ema=100,
        rsi_period=14,
        atr_period=14,
        volume_period=20,
    )

    reclaim_long = (
        (out["close"] > out["ema_fast"])
        & (out["close"].shift(1) <= out["ema_fast"].shift(1))
        & (out["ema_fast"] > out["ema_slow"])
        & (out["close"] > out["trend_ema"])
        & (out["rsi"] >= 52)
        & (out["volume_ratio"] >= 1.2)
    )
    reclaim_short = (
        (out["close"] < out["ema_fast"])
        & (out["close"].shift(1) >= out["ema_fast"].shift(1))
        & (out["ema_fast"] < out["ema_slow"])
        & (out["close"] < out["trend_ema"])
        & (out["rsi"] <= 48)
        & (out["volume_ratio"] >= 1.2)
    )

    out["signal"] = 0
    out.loc[reclaim_long, "signal"] = 1
    out.loc[reclaim_short, "signal"] = -1
    return out


def _mean_reversion(df: pd.DataFrame) -> pd.DataFrame:
    out = add_indicators(
        df,
        ema_fast=9,
        ema_slow=21,
        trend_ema=100,
        rsi_period=14,
        atr_period=14,
        volume_period=20,
    )
    return generate_mean_reversion_variant(out, "Extreme entry")


STRATEGIES = {
    "Vol+Volume Breakout": {
        "builder": _vol_volume_breakout,
        "stop_atr": 1.0,
        "take_atr": 1.5,
    },
    "Momentum Pullback": {
        "builder": _momentum_pullback,
        "stop_atr": 1.0,
        "take_atr": 1.5,
    },
    "Mean Reversion": {
        "builder": _mean_reversion,
        "stop_atr": 1.0,
        "take_atr": 1.25,
    },
}


def _safe_pf(value: float) -> float:
    return 5.0 if math.isinf(float(value)) else float(value)


def run_dynamic_universe_scalping(
    rolling_details: pd.DataFrame,
    *,
    forward_days: int,
    fee_bps: float,
    slippage_bps: float,
    max_periods: int | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Backtest fixed 5m strategies on symbols selected by the rolling universe.

    Each symbol-period is equally weighted inside the weekly portfolio.
    The selection itself is assumed to have been created without future data.
    """
    if rolling_details is None or rolling_details.empty:
        raise ValueError("Run Rolling Universe Validation first.")

    selected = rolling_details[["Period", "Selection time", "Symbol"]].drop_duplicates().copy()
    selected["Selection time"] = pd.to_datetime(selected["Selection time"], utc=True)

    if max_periods is not None:
        keep_periods = sorted(selected["Period"].unique())[-int(max_periods):]
        selected = selected[selected["Period"].isin(keep_periods)]

    data_cache: dict[tuple[str, pd.Timestamp], pd.DataFrame] = {}
    detail_rows = []
    failed_pairs = 0

    for _, sel in selected.iterrows():
        period = int(sel["Period"])
        selection_time = pd.Timestamp(sel["Selection time"])
        symbol = str(sel["Symbol"])
        forward_end = selection_time + pd.Timedelta(days=int(forward_days))
        key = (symbol, selection_time)

        try:
            market = _fetch_5m(symbol, selection_time, forward_end)
        except Exception:
            market = pd.DataFrame()

        data_cache[key] = market
        if market.empty or len(market) < 200:
            failed_pairs += 1
            continue

        for strategy_name, cfg in STRATEGIES.items():
            signal_df = cfg["builder"](market)

            gross = run_backtest(
                signal_df,
                start_cash=10_000.0,
                fee_bps=0.0,
                slippage_bps=0.0,
                use_stop_loss=True,
                stop_atr=float(cfg["stop_atr"]),
                use_take_profit=True,
                take_atr=float(cfg["take_atr"]),
            )
            net = run_backtest(
                signal_df,
                start_cash=10_000.0,
                fee_bps=float(fee_bps),
                slippage_bps=float(slippage_bps),
                use_stop_loss=True,
                stop_atr=float(cfg["stop_atr"]),
                use_take_profit=True,
                take_atr=float(cfg["take_atr"]),
            )

            gm = gross["metrics"]
            nm = net["metrics"]

            detail_rows.append(
                {
                    "Period": period,
                    "Selection time": selection_time,
                    "Symbol": symbol,
                    "Strategy": strategy_name,
                    "Gross return %": float(gm["net_return_pct"]),
                    "Net return %": float(nm["net_return_pct"]),
                    "Cost drag pp": float(gm["net_return_pct"]) - float(nm["net_return_pct"]),
                    "Gross PF": float(gm["profit_factor"]),
                    "Net PF": float(nm["profit_factor"]),
                    "Trades": int(nm["trades"]),
                    "Win rate %": float(nm["win_rate_pct"]),
                    "Max DD %": float(nm["max_drawdown_pct"]),
                    "Long P&L": float(net["long"]["pnl"]),
                    "Short P&L": float(net["short"]["pnl"]),
                }
            )

    details = pd.DataFrame(detail_rows)
    if details.empty:
        raise RuntimeError("No selected symbol-period had enough 5m data for the scalping test.")

    period_rows = []
    for (strategy, period), group in details.groupby(["Strategy", "Period"], sort=True):
        symbol_count = int(group["Symbol"].nunique())
        period_rows.append(
            {
                "Strategy": strategy,
                "Period": int(period),
                "Selection time": group["Selection time"].min(),
                "Symbols": symbol_count,
                "Gross portfolio return %": float(group["Gross return %"].mean()),
                "Net portfolio return %": float(group["Net return %"].mean()),
                "Cost drag pp": float(group["Cost drag pp"].mean()),
                "Trades": int(group["Trades"].sum()),
                "Avg win rate %": float(group["Win rate %"].mean()),
                "Avg max DD %": float(group["Max DD %"].mean()),
            }
        )

    periods = pd.DataFrame(period_rows)

    summary_rows = []
    for strategy, group in periods.groupby("Strategy", sort=False):
        net = group["Net portfolio return %"].astype(float)
        gross = group["Gross portfolio return %"].astype(float)
        compounded = (float((1.0 + net / 100.0).prod()) - 1.0) * 100.0
        gross_compounded = (float((1.0 + gross / 100.0).prod()) - 1.0) * 100.0

        strategy_details = details[details["Strategy"] == strategy]
        pf_values = strategy_details["Net PF"].map(_safe_pf)

        summary_rows.append(
            {
                "Strategy": strategy,
                "Periods": int(len(group)),
                "Positive net periods": f"{int((net > 0).sum())}/{len(group)}",
                "Avg gross return %": float(gross.mean()),
                "Avg net return %": float(net.mean()),
                "Median net return %": float(net.median()),
                "Compounded gross %": gross_compounded,
                "Compounded net %": compounded,
                "Avg cost drag pp": float((gross - net).mean()),
                "Worst period %": float(net.min()),
                "Best period %": float(net.max()),
                "Total trades": int(group["Trades"].sum()),
                "Avg net PF": float(pf_values.mean()),
            }
        )

    summary = pd.DataFrame(summary_rows)
    summary.attrs["failed_pairs"] = int(failed_pairs)
    return summary, periods, details
