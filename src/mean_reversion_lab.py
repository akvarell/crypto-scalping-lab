from __future__ import annotations

import math
import pandas as pd

from src.backtest import run_backtest
from src.indicators import add_indicators
from src.strategy import generate_mean_reversion_variant


VARIANTS = [
    {"variant": "Extreme entry", "stop_atr": 1.00, "take_atr": 1.25},
    {"variant": "Reclaim entry", "stop_atr": 1.00, "take_atr": 1.25},
    {"variant": "Reclaim + flat regime", "stop_atr": 1.00, "take_atr": 1.25},
    {"variant": "Reclaim + flat + vol guard", "stop_atr": 1.00, "take_atr": 1.25},
]


def evaluate_mean_reversion_variants(
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
    folds = max(2, min(int(folds), 6))
    calibration_pct = max(20, min(int(calibration_pct), 70))

    n = len(raw_df)
    calibration_end = int(n * calibration_pct / 100)
    calibration_end = max(1, min(calibration_end, n - folds))

    remaining = n - calibration_end
    fold_size = remaining // folds
    if fold_size < 20:
        raise ValueError("Not enough candles for the requested mean-reversion folds.")

    summary_rows = []
    detail_rows = []

    for cfg in VARIANTS:
        work = add_indicators(
            raw_df,
            ema_fast=9,
            ema_slow=21,
            trend_ema=100,
            rsi_period=int(rsi_period),
            atr_period=int(atr_period),
        )
        work = generate_mean_reversion_variant(work, cfg["variant"])

        returns = []
        pfs = []
        dds = []
        total_trades = 0
        positive_folds = 0
        low_sample_folds = 0
        long_pnl = 0.0
        short_pnl = 0.0
        long_trades = 0
        short_trades = 0

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
                stop_atr=float(cfg["stop_atr"]),
                use_take_profit=True,
                take_atr=float(cfg["take_atr"]),
            )
            m = result["metrics"]
            ret = float(m["net_return_pct"])
            pf = float(m["profit_factor"])
            dd = float(m["max_drawdown_pct"])
            trades = int(m["trades"])

            returns.append(ret)
            pfs.append(5.0 if math.isinf(pf) else pf)
            dds.append(dd)
            total_trades += trades
            positive_folds += int(ret > 0)
            low_sample_folds += int(trades < int(min_fold_trades))

            long_pnl += float(result["long"]["pnl"])
            short_pnl += float(result["short"]["pnl"])
            long_trades += int(result["long"]["trades"])
            short_trades += int(result["short"]["trades"])

            detail_rows.append(
                {
                    "Variant": cfg["variant"],
                    "Fold": fold + 1,
                    "Start": fold_df.index[0] if not fold_df.empty else None,
                    "End": fold_df.index[-1] if not fold_df.empty else None,
                    "Return %": ret,
                    "PF": pf,
                    "DD %": dd,
                    "Trades": trades,
                    "Long P&L": float(result["long"]["pnl"]),
                    "Short P&L": float(result["short"]["pnl"]),
                    "Sample": "OK" if trades >= int(min_fold_trades) else "LOW SAMPLE",
                }
            )

        ret_s = pd.Series(returns, dtype=float)
        pf_s = pd.Series(pfs, dtype=float)
        dd_s = pd.Series(dds, dtype=float)

        summary_rows.append(
            {
                "Variant": cfg["variant"],
                "Positive folds": f"{positive_folds}/{folds}",
                "Median return %": float(ret_s.median()),
                "Average return %": float(ret_s.mean()),
                "Worst fold %": float(ret_s.min()),
                "Best fold %": float(ret_s.max()),
                "Average PF": float(pf_s.mean()),
                "Worst DD %": float(dd_s.min()),
                "Total trades": int(total_trades),
                "Long trades": int(long_trades),
                "Long P&L": float(long_pnl),
                "Short trades": int(short_trades),
                "Short P&L": float(short_pnl),
                "Low-sample folds": int(low_sample_folds),
            }
        )

    return pd.DataFrame(summary_rows), pd.DataFrame(detail_rows)
