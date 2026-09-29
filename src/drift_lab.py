from __future__ import annotations

import pandas as pd

from src.entry_quality_holdout import _evaluate_frozen_entries, run_frozen_1m_entry_holdout
from src.feature_drift import compare_feature_drift


def run_regime_feature_drift_lab(
    development_rolling_details: pd.DataFrame,
    *,
    development_periods: int = 12,
    holdout_days: int = 120,
    holdout_end_offset_days: int = 270,
    lookback_days: int = 14,
    forward_days: int = 7,
    pool_size: int = 20,
    select_top_n: int = 5,
    min_daily_turnover_usd: float = 5_000_000.0,
    fee_bps: float = 4.0,
    slippage_bps: float = 2.0,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Regenerate comparable recent and older event samples, then diagnose drift.

    This is descriptive research. It does not tune thresholds or create a new
    trading rule from the older holdout.
    """
    if development_rolling_details is None or development_rolling_details.empty:
        raise ValueError("Run Rolling Universe Validation first.")

    dev = development_rolling_details.copy()
    keep_periods = sorted(dev["Period"].unique())[-int(development_periods):]
    dev = dev[dev["Period"].isin(keep_periods)].copy()

    dev_summary, dev_details = _evaluate_frozen_entries(
        dev,
        forward_days=int(forward_days),
        fee_bps=float(fee_bps),
        slippage_bps=float(slippage_bps),
    )

    hold_summary, hold_details, _, _ = run_frozen_1m_entry_holdout(
        horizon_days=int(holdout_days),
        end_offset_days=int(holdout_end_offset_days),
        lookback_days=int(lookback_days),
        forward_days=int(forward_days),
        pool_size=int(pool_size),
        select_top_n=int(select_top_n),
        min_daily_turnover_usd=float(min_daily_turnover_usd),
        fee_bps=float(fee_bps),
        slippage_bps=float(slippage_bps),
    )

    edge, drift, corr = compare_feature_drift(dev_details, hold_details)

    return edge, drift, corr, dev_summary, hold_summary
