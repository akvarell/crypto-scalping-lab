from __future__ import annotations

import time
from typing import Callable

import pandas as pd

from src.continuation_exhaustion import analyze_continuation_exhaustion
from src.entry_quality_holdout import _evaluate_frozen_entries, run_frozen_1m_entry_holdout
from src.feature_drift import compare_feature_drift
from src.rolling_universe import RESEARCH_ANCHOR_UTC, run_rolling_universe_validation


ProgressCallback = Callable[[int, int, str], None]
SummaryCallback = Callable[[int, int, str, str, str], None]


def _notify(callback: ProgressCallback | None, step: int, total: int, label: str) -> None:
    if callback is not None:
        callback(step, total, label)


def _notify_summary(
    callback: SummaryCallback | None,
    step: int,
    total: int,
    label: str,
    status: str,
    message: str,
) -> None:
    if callback is not None:
        callback(step, total, label, status, message)


def _tier_row(frame: pd.DataFrame, tier: str) -> pd.Series | None:
    if frame is None or frame.empty or "Tier" not in frame:
        return None
    match = frame[frame["Tier"].astype(str) == tier]
    return match.iloc[0] if not match.empty else None


def _period_fraction(value: object) -> tuple[int, int]:
    try:
        left, right = str(value).split("/", 1)
        return int(left), int(right)
    except Exception:
        return 0, 0


def _stage_summary_universe(summary: dict) -> tuple[str, str]:
    periods = int(summary.get("periods", 0))
    positive = int(summary.get("positive_vol_uplift_periods", 0))
    uplift = float(summary.get("avg_vol_uplift_pct", 0.0))
    move = float(summary.get("avg_move_uplift_pct", 0.0))

    if periods > 0 and positive == periods and uplift > 0:
        status = "PASS"
        text = (
            f"Universe selection stayed positive in {positive}/{periods} periods. "
            f"Average volatility uplift was {uplift:.1f}% and average move uplift was {move:.1f}%. "
            "The screener remains the strongest validated component; do not retune it from entry-strategy results."
        )
    else:
        status = "REVIEW"
        text = (
            f"Universe selection was positive in {positive}/{periods} periods with "
            f"{uplift:.1f}% average volatility uplift. The screener is not uniformly stable in this run, "
            "so entry-strategy conclusions should be treated cautiously."
        )
    return status, text


def _stage_summary_development(summary: pd.DataFrame) -> tuple[str, str]:
    strong = _tier_row(summary, "Strong 1m confirm")
    dual = _tier_row(summary, "Dual strong")

    parts = []
    any_candidate = False
    for label, row in [("Strong 1m", strong), ("Dual strong", dual)]:
        if row is None:
            continue
        net = float(row.get("Net avg bps", 0.0))
        median = float(row.get("Net median bps", 0.0))
        trimmed = float(row.get("Trimmed gross avg bps", 0.0))
        events = int(row.get("Events", 0))
        periods = int(row.get("Periods", 0))
        parts.append(
            f"{label}: {events} events/{periods} periods, net avg {net:.1f} bps, "
            f"net median {median:.1f} bps, trimmed gross {trimmed:.1f} bps."
        )
        if net > 0 and trimmed > 12.0 and events >= 20 and periods >= 6:
            any_candidate = True

    if any_candidate:
        status = "CANDIDATE"
        suffix = (
            "At least one subset looks interesting on development data, but it is not validated until the frozen holdout."
        )
    else:
        status = "WEAK"
        suffix = (
            "No subset clears a strong development bar after costs and robust averaging; do not manufacture a candidate by tuning thresholds."
        )

    return status, " ".join(parts + [suffix])


def _stage_summary_holdout(summary: pd.DataFrame) -> tuple[str, str]:
    strong = _tier_row(summary, "Strong 1m confirm")
    dual = _tier_row(summary, "Dual strong")
    survivors = []
    parts = []

    for label, row in [("Strong 1m", strong), ("Dual strong", dual)]:
        if row is None:
            continue
        net = float(row.get("Net avg bps", 0.0))
        median = float(row.get("Net median bps", 0.0))
        trimmed = float(row.get("Trimmed gross avg bps", 0.0))
        events = int(row.get("Events", 0))
        positive, total = _period_fraction(row.get("Positive period avg", "0/0"))

        parts.append(
            f"{label}: {events} events, net avg {net:.1f} bps, net median {median:.1f} bps, "
            f"trimmed gross {trimmed:.1f} bps, positive periods {positive}/{total}."
        )

        if (
            net > 0
            and median > 0
            and trimmed > 12.0
            and total > 0
            and positive > total / 2
            and events >= 20
        ):
            survivors.append(label)

    if survivors:
        status = "SURVIVED"
        suffix = (
            "Frozen evidence survived for: " + ", ".join(survivors) + ". "
            "Do not retune it here; the next validation must use an untouched window."
        )
    else:
        status = "FAILED"
        suffix = (
            "The frozen entry hypothesis did not survive the independent window. "
            "Do not optimize its thresholds on this holdout; use drift diagnostics to form a genuinely new hypothesis."
        )

    return status, " ".join(parts + [suffix])


def _stage_summary_drift(
    edge: pd.DataFrame,
    drift: pd.DataFrame,
    corr: pd.DataFrame,
) -> tuple[str, str]:
    large_shifts = pd.DataFrame()
    if drift is not None and not drift.empty:
        large_shifts = drift[drift["Robust shift"].abs() >= 0.5].copy()

    flips = pd.DataFrame()
    if corr is not None and not corr.empty:
        flips = corr[corr["Same direction"].astype(str) == "NO"].copy()

    shift_names = []
    if not large_shifts.empty:
        ranked = large_shifts.assign(_abs=large_shifts["Robust shift"].abs()).sort_values(
            "_abs", ascending=False
        )
        shift_names = ranked["Feature"].drop_duplicates().head(3).tolist()

    flip_names = []
    if not flips.empty:
        flip_names = flips["Feature"].drop_duplicates().head(3).tolist()

    if shift_names or flip_names:
        status = "REGIME_DRIFT"
        bits = []
        if shift_names:
            bits.append("largest distribution shifts: " + ", ".join(shift_names))
        if flip_names:
            bits.append("feature/return direction changed for: " + ", ".join(flip_names))
        text = (
            "The development and older samples are not behaving as the same regime; "
            + "; ".join(bits)
            + ". Treat the previous development edge as regime-specific until a new economic hypothesis is frozen."
        )
    else:
        status = "SIGNAL_INSTABILITY"
        text = (
            "The measured features did not show a large enough distribution/relationship shift to explain the failure. "
            "That points more toward unstable entry logic or omitted variables than a simple threshold drift."
        )

    return status, text


def run_research_pipeline(
    *,
    horizon_days: int = 90,
    lookback_days: int = 14,
    forward_days: int = 7,
    pool_size: int = 20,
    select_top_n: int = 5,
    min_daily_turnover_usd: float = 5_000_000.0,
    fee_bps: float = 4.0,
    slippage_bps: float = 2.0,
    holdout_days: int = 120,
    holdout_end_offset_days: int = 270,
    development_periods: int = 12,
    progress_callback: ProgressCallback | None = None,
    summary_callback: SummaryCallback | None = None,
) -> dict:
    total_steps = 5
    timings = {}
    summaries = []

    _notify(progress_callback, 1, total_steps, "Rolling Universe Validation")
    started = time.perf_counter()
    rolling_periods, rolling_details, rolling_summary = run_rolling_universe_validation(
        horizon_days=int(horizon_days),
        lookback_days=int(lookback_days),
        forward_days=int(forward_days),
        pool_size=int(pool_size),
        select_top_n=int(select_top_n),
        min_daily_turnover_usd=float(min_daily_turnover_usd),
        end_offset_days=0,
        as_of=RESEARCH_ANCHOR_UTC,
    )
    timings["Rolling Universe"] = time.perf_counter() - started
    status, message = _stage_summary_universe(rolling_summary)
    summaries.append(
        {"Step": 1, "Test": "Rolling Universe", "Status": status, "What became clear": message}
    )
    _notify_summary(summary_callback, 1, total_steps, "Rolling Universe", status, message)

    if rolling_details is None or rolling_details.empty:
        raise RuntimeError("Rolling Universe returned no historical selections.")

    _notify(progress_callback, 2, total_steps, "Frozen-format development entry study")
    started = time.perf_counter()
    recent = rolling_details.copy()
    keep_periods = sorted(recent["Period"].unique())[-int(development_periods):]
    recent = recent[recent["Period"].isin(keep_periods)].copy()
    development_summary, development_details = _evaluate_frozen_entries(
        recent,
        forward_days=int(forward_days),
        fee_bps=float(fee_bps),
        slippage_bps=float(slippage_bps),
    )
    timings["Development Entry Study"] = time.perf_counter() - started
    status, message = _stage_summary_development(development_summary)
    summaries.append(
        {"Step": 2, "Test": "Development Entry Study", "Status": status, "What became clear": message}
    )
    _notify_summary(summary_callback, 2, total_steps, "Development Entry Study", status, message)

    _notify(progress_callback, 3, total_steps, "Frozen 1m Entry Holdout")
    started = time.perf_counter()
    holdout_summary, holdout_details, holdout_periods, holdout_universe_summary = run_frozen_1m_entry_holdout(
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
    timings["Frozen Holdout"] = time.perf_counter() - started
    status, message = _stage_summary_holdout(holdout_summary)
    summaries.append(
        {"Step": 3, "Test": "Frozen 1m Holdout", "Status": status, "What became clear": message}
    )
    _notify_summary(summary_callback, 3, total_steps, "Frozen 1m Holdout", status, message)

    _notify(progress_callback, 4, total_steps, "Regime & Feature Drift")
    started = time.perf_counter()
    drift_edge, feature_drift, feature_corr = compare_feature_drift(
        development_details,
        holdout_details,
    )
    timings["Regime & Feature Drift"] = time.perf_counter() - started
    status, message = _stage_summary_drift(drift_edge, feature_drift, feature_corr)
    summaries.append(
        {"Step": 4, "Test": "Regime & Feature Drift", "Status": status, "What became clear": message}
    )
    _notify_summary(summary_callback, 4, total_steps, "Regime & Feature Drift", status, message)

    _notify(progress_callback, 5, total_steps, "Continuation vs Exhaustion")
    started = time.perf_counter()
    continuation_summary, continuation_relationships, continuation_verdict = (
        analyze_continuation_exhaustion(development_details)
    )
    timings["Continuation vs Exhaustion"] = time.perf_counter() - started

    coherent_count = int(
        (
            continuation_summary["Coherent diagnostic"].astype(str) == "YES"
        ).sum()
    )
    continuation_status = (
        "HYPOTHESIS_FOUND" if coherent_count > 0 else "NO_CLEAR_SPLIT"
    )
    summaries.append(
        {
            "Step": 5,
            "Test": "Continuation vs Exhaustion",
            "Status": continuation_status,
            "What became clear": continuation_verdict,
        }
    )
    _notify_summary(
        summary_callback,
        5,
        total_steps,
        "Continuation vs Exhaustion",
        continuation_status,
        continuation_verdict,
    )

    summary_table = pd.DataFrame(summaries)
    timing_table = pd.DataFrame(
        [
            {"Test": key, "Seconds": value, "Minutes": value / 60.0}
            for key, value in timings.items()
        ]
    )

    next_allowed_test = (
        "Use v2.6 only to formulate ONE simple continuation-vs-exhaustion hypothesis. "
        "Freeze that rule before opening the third untouched 120-day window ending 390 days "
        "before the frozen research anchor. Do not retune it on the v2.3 window."
    )

    return {
        "rolling_periods": rolling_periods,
        "rolling_details": rolling_details,
        "rolling_summary": rolling_summary,
        "development_summary": development_summary,
        "development_details": development_details,
        "holdout_summary": holdout_summary,
        "holdout_details": holdout_details,
        "holdout_periods": holdout_periods,
        "holdout_universe_summary": holdout_universe_summary,
        "drift_edge": drift_edge,
        "feature_drift": feature_drift,
        "feature_corr": feature_corr,
        "continuation_summary": continuation_summary,
        "continuation_relationships": continuation_relationships,
        "continuation_verdict": continuation_verdict,
        "stage_summaries": summary_table,
        "timings": timing_table,
        "next_allowed_test": next_allowed_test,
        "research_anchor": RESEARCH_ANCHOR_UTC.isoformat(),
    }
