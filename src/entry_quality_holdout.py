from __future__ import annotations

import pandas as pd

from src.entry_quality_lab import run_entry_quality_lab
from src.rolling_universe import run_rolling_universe_validation


FROZEN_TIERS = [
    "Baseline LONG/BULL",
    "Strong 1m confirm",
    "Dual strong",
]


def run_frozen_1m_entry_holdout(
    *,
    horizon_days: int = 120,
    end_offset_days: int = 270,
    lookback_days: int = 14,
    forward_days: int = 7,
    pool_size: int = 20,
    select_top_n: int = 5,
    min_daily_turnover_usd: float = 5_000_000.0,
    fee_bps: float = 4.0,
    slippage_bps: float = 2.0,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, dict]:
    """Older, non-overlapping holdout for frozen v2.2 entry-quality hypotheses.

    Frozen before this window is examined:
      * Baseline LONG/BULL control
      * Strong 1m confirm
      * Dual strong
      * fixed 10-minute exit
      * 4 bps fee + 2 bps slippage per side by default

    The window ends end_offset_days before today, keeping it separate from the
    recent development period and the previously examined v1.9 holdout.
    """
    rolling_periods, rolling_details, rolling_summary = run_rolling_universe_validation(
        horizon_days=int(horizon_days),
        lookback_days=int(lookback_days),
        forward_days=int(forward_days),
        pool_size=int(pool_size),
        select_top_n=int(select_top_n),
        min_daily_turnover_usd=float(min_daily_turnover_usd),
        end_offset_days=int(end_offset_days),
    )

    if rolling_details is None or rolling_details.empty:
        raise RuntimeError("Older holdout universe produced no selected symbols.")

    full_summary, full_details = run_entry_quality_lab(
        rolling_details,
        forward_days=int(forward_days),
        fee_bps=float(fee_bps),
        slippage_bps=float(slippage_bps),
        max_periods=999,
    )

    summary = full_summary[full_summary["Tier"].isin(FROZEN_TIERS)].copy()
    details = full_details[full_details["Tier"].isin(FROZEN_TIERS)].copy()

    order = {name: i for i, name in enumerate(FROZEN_TIERS)}
    summary["_order"] = summary["Tier"].map(order)
    summary = summary.sort_values("_order").drop(columns=["_order"]).reset_index(drop=True)

    return summary, details, rolling_periods, rolling_summary
