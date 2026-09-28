from __future__ import annotations

from itertools import product
import math
import pandas as pd

from src.backtest import run_backtest
from src.indicators import add_indicators
from src.strategy import generate_signals


def _score(metrics: dict, objective: str, min_train_trades: int) -> float:
    trades = int(metrics["trades"])
    if trades < int(min_train_trades):
        return -1_000_000.0 + trades

    ret = float(metrics["net_return_pct"])
    dd = abs(float(metrics["max_drawdown_pct"]))
    pf_raw = float(metrics["profit_factor"])
    pf = 5.0 if math.isinf(pf_raw) else min(pf_raw, 5.0)

    if objective == "Net return":
        return ret - 0.20 * dd
    if objective == "Profit factor":
        return pf + 0.02 * ret - 0.03 * dd

    # Balanced is a research heuristic, not a trading recommendation.
    return ret - 0.70 * dd + 0.80 * (pf - 1.0) + 0.005 * min(trades, 100)


def optimize_quick(
    raw_df: pd.DataFrame,
    split_idx: int,
    *,
    objective: str,
    rsi_period: int,
    atr_period: int,
    use_trend_filter: bool,
    use_volume_filter: bool,
    min_volume_ratio: float,
    start_cash: float,
    fee_bps: float,
    slippage_bps: float,
    min_train_trades: int = 20,
    min_test_trades: int = 10,
    top_n: int = 8,
) -> pd.DataFrame:
    """Rank on TRAIN only and report TEST without using TEST for ranking."""

    fast_values = [5, 9, 13]
    slow_values = [21, 34, 55]
    rsi_pairs = [(50, 50), (52, 48), (55, 45)]
    risk_pairs = [(0.75, 1.00), (1.00, 1.50), (1.25, 2.00), (1.50, 2.50)]
    trend_values = [50, 100, 200] if use_trend_filter else [100]

    train_rows = []

    for ema_fast, ema_slow, rsi_pair, risk_pair, trend_ema in product(
        fast_values,
        slow_values,
        rsi_pairs,
        risk_pairs,
        trend_values,
    ):
        if ema_fast >= ema_slow:
            continue

        rsi_long, rsi_short = rsi_pair
        stop_atr, take_atr = risk_pair

        work = add_indicators(
            raw_df,
            ema_fast=ema_fast,
            ema_slow=ema_slow,
            trend_ema=trend_ema,
            rsi_period=int(rsi_period),
            atr_period=int(atr_period),
        )
        work = generate_signals(
            work,
            rsi_long=rsi_long,
            rsi_short=rsi_short,
            use_trend_filter=bool(use_trend_filter),
            use_volume_filter=bool(use_volume_filter),
            min_volume_ratio=float(min_volume_ratio),
        )

        train_df = work.iloc[:split_idx].copy()
        result = run_backtest(
            train_df,
            start_cash=float(start_cash),
            fee_bps=float(fee_bps),
            slippage_bps=float(slippage_bps),
            use_stop_loss=True,
            stop_atr=float(stop_atr),
            use_take_profit=True,
            take_atr=float(take_atr),
        )
        m = result["metrics"]
        train_trades = int(m["trades"])

        train_rows.append(
            {
                "EMA fast": ema_fast,
                "EMA slow": ema_slow,
                "Trend EMA": trend_ema,
                "RSI long": rsi_long,
                "RSI short": rsi_short,
                "Stop ATR": stop_atr,
                "Take ATR": take_atr,
                "Train return %": float(m["net_return_pct"]),
                "Train PF": float(m["profit_factor"]),
                "Train DD %": float(m["max_drawdown_pct"]),
                "Train trades": train_trades,
                "Train sample": "OK" if train_trades >= int(min_train_trades) else "LOW SAMPLE",
                "Score": _score(m, objective, int(min_train_trades)),
            }
        )

    ranked = pd.DataFrame(train_rows).sort_values(
        ["Score", "Train return %"],
        ascending=[False, False],
    )
    ranked = ranked.head(int(top_n)).reset_index(drop=True)

    test_rows = []
    for _, row in ranked.iterrows():
        work = add_indicators(
            raw_df,
            ema_fast=int(row["EMA fast"]),
            ema_slow=int(row["EMA slow"]),
            trend_ema=int(row["Trend EMA"]),
            rsi_period=int(rsi_period),
            atr_period=int(atr_period),
        )
        work = generate_signals(
            work,
            rsi_long=int(row["RSI long"]),
            rsi_short=int(row["RSI short"]),
            use_trend_filter=bool(use_trend_filter),
            use_volume_filter=bool(use_volume_filter),
            min_volume_ratio=float(min_volume_ratio),
        )
        test_df = work.iloc[split_idx:].copy()
        result = run_backtest(
            test_df,
            start_cash=float(start_cash),
            fee_bps=float(fee_bps),
            slippage_bps=float(slippage_bps),
            use_stop_loss=True,
            stop_atr=float(row["Stop ATR"]),
            use_take_profit=True,
            take_atr=float(row["Take ATR"]),
        )
        m = result["metrics"]
        test_trades = int(m["trades"])
        test_rows.append(
            {
                "Test return %": float(m["net_return_pct"]),
                "Test PF": float(m["profit_factor"]),
                "Test DD %": float(m["max_drawdown_pct"]),
                "Test trades": test_trades,
                "Test win %": float(m["win_rate_pct"]),
                "Test sample": "OK" if test_trades >= int(min_test_trades) else "LOW SAMPLE",
            }
        )

    return pd.concat([ranked, pd.DataFrame(test_rows)], axis=1)
