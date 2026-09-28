from __future__ import annotations

import math
import pandas as pd

from src.backtest import run_backtest
from src.indicators import add_indicators
from src.strategy import generate_mean_reversion_variant


def _pf_cap(value: float) -> float:
    return 5.0 if math.isinf(float(value)) else float(value)


def evaluate_edge_diagnostics(
    raw_df: pd.DataFrame,
    *,
    folds: int,
    calibration_pct: int,
    rsi_period: int,
    atr_period: int,
    start_cash: float,
    fee_bps: float,
    slippage_bps: float,
    min_fold_trades: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Diagnose gross-vs-net and LONG-vs-SHORT edge for the fixed MR baseline.

    Uses the v0.7 Extreme-entry Mean Reversion reference without retuning it.
    """
    folds = max(2, min(int(folds), 6))
    calibration_pct = max(20, min(int(calibration_pct), 70))

    n = len(raw_df)
    calibration_end = int(n * calibration_pct / 100)
    calibration_end = max(1, min(calibration_end, n - folds))
    remaining = n - calibration_end
    fold_size = remaining // folds
    if fold_size < 20:
        raise ValueError("Not enough candles for the requested diagnostics.")

    work = add_indicators(
        raw_df,
        ema_fast=9,
        ema_slow=21,
        trend_ema=100,
        rsi_period=int(rsi_period),
        atr_period=int(atr_period),
    )
    work = generate_mean_reversion_variant(work, "Extreme entry")

    scopes = {
        "Both sides": None,
        "LONG only": 1,
        "SHORT only": -1,
    }

    summary_rows = []
    detail_rows = []

    for scope_name, allowed_signal in scopes.items():
        scoped = work.copy()
        if allowed_signal == 1:
            scoped.loc[scoped["signal"] < 0, "signal"] = 0
        elif allowed_signal == -1:
            scoped.loc[scoped["signal"] > 0, "signal"] = 0

        gross_returns = []
        net_returns = []
        gross_pfs = []
        net_pfs = []
        net_dds = []
        gross_positive = 0
        net_positive = 0
        total_trades = 0
        low_sample_folds = 0

        for fold in range(folds):
            start = calibration_end + fold * fold_size
            end = calibration_end + (fold + 1) * fold_size
            if fold == folds - 1:
                end = n

            fold_df = scoped.iloc[start:end].copy()

            gross = run_backtest(
                fold_df,
                start_cash=float(start_cash),
                fee_bps=0.0,
                slippage_bps=0.0,
                use_stop_loss=True,
                stop_atr=1.0,
                use_take_profit=True,
                take_atr=1.25,
            )
            net = run_backtest(
                fold_df,
                start_cash=float(start_cash),
                fee_bps=float(fee_bps),
                slippage_bps=float(slippage_bps),
                use_stop_loss=True,
                stop_atr=1.0,
                use_take_profit=True,
                take_atr=1.25,
            )

            gm = gross["metrics"]
            nm = net["metrics"]
            trades = int(nm["trades"])

            gross_return = float(gm["net_return_pct"])
            net_return = float(nm["net_return_pct"])
            gross_pf = float(gm["profit_factor"])
            net_pf = float(nm["profit_factor"])

            gross_returns.append(gross_return)
            net_returns.append(net_return)
            gross_pfs.append(_pf_cap(gross_pf))
            net_pfs.append(_pf_cap(net_pf))
            net_dds.append(float(nm["max_drawdown_pct"]))
            gross_positive += int(gross_return > 0)
            net_positive += int(net_return > 0)
            total_trades += trades
            low_sample_folds += int(trades < int(min_fold_trades))

            detail_rows.append(
                {
                    "Scope": scope_name,
                    "Fold": fold + 1,
                    "Start": fold_df.index[0] if not fold_df.empty else None,
                    "End": fold_df.index[-1] if not fold_df.empty else None,
                    "Gross return %": gross_return,
                    "Net return %": net_return,
                    "Cost drag pp": gross_return - net_return,
                    "Gross PF": gross_pf,
                    "Net PF": net_pf,
                    "Net DD %": float(nm["max_drawdown_pct"]),
                    "Trades": trades,
                    "Sample": "OK" if trades >= int(min_fold_trades) else "LOW SAMPLE",
                }
            )

        gross_s = pd.Series(gross_returns, dtype=float)
        net_s = pd.Series(net_returns, dtype=float)
        gross_pf_s = pd.Series(gross_pfs, dtype=float)
        net_pf_s = pd.Series(net_pfs, dtype=float)
        net_dd_s = pd.Series(net_dds, dtype=float)

        summary_rows.append(
            {
                "Scope": scope_name,
                "Gross positive folds": f"{gross_positive}/{folds}",
                "Net positive folds": f"{net_positive}/{folds}",
                "Gross avg return %": float(gross_s.mean()),
                "Net avg return %": float(net_s.mean()),
                "Net median return %": float(net_s.median()),
                "Cost drag pp": float((gross_s - net_s).mean()),
                "Gross avg PF": float(gross_pf_s.mean()),
                "Net avg PF": float(net_pf_s.mean()),
                "Net worst fold %": float(net_s.min()),
                "Net worst DD %": float(net_dd_s.min()),
                "Total trades": int(total_trades),
                "Low-sample folds": int(low_sample_folds),
            }
        )

    return pd.DataFrame(summary_rows), pd.DataFrame(detail_rows)
