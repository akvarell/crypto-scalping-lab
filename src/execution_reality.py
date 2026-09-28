from __future__ import annotations

import math
import pandas as pd

from src.dynamic_scalping import STRATEGIES, _fetch_5m
from src.execution_backtest import run_strict_backtest


CONFIGS = [
    {"label": "Both · no cooldown", "side": "BOTH", "cooldown": 0},
    {"label": "LONG only · no cooldown", "side": "LONG", "cooldown": 0},
    {"label": "SHORT only · no cooldown", "side": "SHORT", "cooldown": 0},
    {"label": "Both · 30m cooldown", "side": "BOTH", "cooldown": 6},
    {"label": "Both · 60m cooldown", "side": "BOTH", "cooldown": 12},
]


def _safe_pf(value: float) -> float:
    return 5.0 if math.isinf(float(value)) else float(value)


def run_execution_reality_check(
    rolling_details: pd.DataFrame,
    *,
    strategy_name: str,
    forward_days: int,
    fee_bps: float,
    slippage_bps: float,
    max_periods: int = 12,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    if rolling_details is None or rolling_details.empty:
        raise ValueError("Run Rolling Universe Validation first.")
    if strategy_name not in STRATEGIES:
        raise ValueError(f"Unknown strategy: {strategy_name}")

    selected = rolling_details[["Period", "Selection time", "Symbol"]].drop_duplicates().copy()
    selected["Selection time"] = pd.to_datetime(selected["Selection time"], utc=True)
    keep_periods = sorted(selected["Period"].unique())[-int(max_periods):]
    selected = selected[selected["Period"].isin(keep_periods)]

    strategy_cfg = STRATEGIES[strategy_name]
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

        signal_df = strategy_cfg["builder"](market)

        for cfg in CONFIGS:
            gross = run_strict_backtest(
                signal_df,
                start_cash=10_000.0,
                fee_bps=0.0,
                slippage_bps=0.0,
                stop_atr=float(strategy_cfg["stop_atr"]),
                take_atr=float(strategy_cfg["take_atr"]),
                side=cfg["side"],
                cooldown_bars=int(cfg["cooldown"]),
            )
            net = run_strict_backtest(
                signal_df,
                start_cash=10_000.0,
                fee_bps=float(fee_bps),
                slippage_bps=float(slippage_bps),
                stop_atr=float(strategy_cfg["stop_atr"]),
                take_atr=float(strategy_cfg["take_atr"]),
                side=cfg["side"],
                cooldown_bars=int(cfg["cooldown"]),
            )

            gm = gross["metrics"]
            nm = net["metrics"]

            rows.append(
                {
                    "Config": cfg["label"],
                    "Period": period,
                    "Selection time": selection_time,
                    "Symbol": symbol,
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
        raise RuntimeError("No 5m symbol-periods were available for the reality check.")

    period_rows = []
    for (config, period), group in details.groupby(["Config", "Period"], sort=True):
        period_rows.append(
            {
                "Config": config,
                "Period": int(period),
                "Selection time": group["Selection time"].min(),
                "Symbols": int(group["Symbol"].nunique()),
                "Gross portfolio return %": float(group["Gross return %"].mean()),
                "Net portfolio return %": float(group["Net return %"].mean()),
                "Cost drag pp": float(group["Cost drag pp"].mean()),
                "Trades": int(group["Trades"].sum()),
                "Avg trade %": float(group["Avg trade %"].mean()),
            }
        )
    periods = pd.DataFrame(period_rows)

    summary_rows = []
    for config, group in periods.groupby("Config", sort=False):
        net = group["Net portfolio return %"].astype(float)
        gross = group["Gross portfolio return %"].astype(float)
        config_details = details[details["Config"] == config]
        net_pf = config_details["Net PF"].map(_safe_pf)

        summary_rows.append(
            {
                "Config": config,
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
                "Avg trade %": float(config_details["Avg trade %"].mean()),
                "Avg net PF": float(net_pf.mean()),
            }
        )

    return pd.DataFrame(summary_rows), periods, details
