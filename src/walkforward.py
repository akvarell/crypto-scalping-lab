from __future__ import annotations

import math
import pandas as pd

from src.backtest import run_backtest
from src.indicators import add_indicators
from src.strategy import generate_signals


def _pf_value(value: float) -> float:
    return 5.0 if math.isinf(float(value)) else float(value)


def evaluate_walk_forward(
    raw_df: pd.DataFrame,
    candidates: pd.DataFrame,
    *,
    folds: int,
    calibration_pct: int,
    rsi_period: int,
    atr_period: int,
    use_trend_filter: bool,
    use_volume_filter: bool,
    min_volume_ratio: float,
    start_cash: float,
    fee_bps: float,
    slippage_bps: float,
    min_fold_trades: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Evaluate already-selected TRAIN candidates over sequential future folds.

    Candidate order is preserved. Walk-forward results are descriptive only and
    are not used to re-rank the candidates.
    """
    folds = max(2, min(int(folds), 6))
    calibration_pct = max(20, min(int(calibration_pct), 70))
    n = len(raw_df)
    calibration_end = int(n * calibration_pct / 100)
    calibration_end = max(1, min(calibration_end, n - folds))

    remaining = n - calibration_end
    fold_size = remaining // folds
    if fold_size < 20:
        raise ValueError("Not enough candles for the requested walk-forward folds.")

    summary_rows = []
    detail_rows = []

    for rank, (_, row) in enumerate(candidates.iterrows(), start=1):
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

        fold_returns = []
        fold_dds = []
        fold_pfs = []
        total_trades = 0
        positive_folds = 0
        low_sample_folds = 0

        for fold in range(folds):
            start = calibration_end + fold * fold_size
            end = calibration_end + (fold + 1) * fold_size
            if fold == folds - 1:
                end = n

            fold_df = work.iloc[start:end].copy()
            result = run_backtest(
                fold_df,
                start_cash=float(start_cash),
                fee_bps=float(fee_bps),
                slippage_bps=float(slippage_bps),
                use_stop_loss=True,
                stop_atr=float(row["Stop ATR"]),
                use_take_profit=True,
                take_atr=float(row["Take ATR"]),
            )
            m = result["metrics"]
            ret = float(m["net_return_pct"])
            dd = float(m["max_drawdown_pct"])
            pf = float(m["profit_factor"])
            trades = int(m["trades"])

            fold_returns.append(ret)
            fold_dds.append(dd)
            fold_pfs.append(_pf_value(pf))
            total_trades += trades
            positive_folds += int(ret > 0)
            low_sample_folds += int(trades < int(min_fold_trades))

            detail_rows.append(
                {
                    "Rank": rank,
                    "Fold": fold + 1,
                    "Start": fold_df.index[0] if not fold_df.empty else None,
                    "End": fold_df.index[-1] if not fold_df.empty else None,
                    "Return %": ret,
                    "PF": pf,
                    "DD %": dd,
                    "Trades": trades,
                    "Sample": "OK" if trades >= int(min_fold_trades) else "LOW SAMPLE",
                }
            )

        returns = pd.Series(fold_returns, dtype=float)
        dds = pd.Series(fold_dds, dtype=float)
        pfs = pd.Series(fold_pfs, dtype=float)

        summary_rows.append(
            {
                "Rank": rank,
                "EMA": f"{int(row['EMA fast'])}/{int(row['EMA slow'])}",
                "Trend EMA": int(row["Trend EMA"]),
                "RSI": f"{int(row['RSI long'])}/{int(row['RSI short'])}",
                "Stop/Take ATR": f"{float(row['Stop ATR']):.2f}/{float(row['Take ATR']):.2f}",
                "Positive folds": f"{positive_folds}/{folds}",
                "Median return %": float(returns.median()),
                "Average return %": float(returns.mean()),
                "Worst fold %": float(returns.min()),
                "Best fold %": float(returns.max()),
                "Average PF": float(pfs.mean()),
                "Worst DD %": float(dds.min()),
                "Total trades": int(total_trades),
                "Low-sample folds": int(low_sample_folds),
            }
        )

    return pd.DataFrame(summary_rows), pd.DataFrame(detail_rows)
