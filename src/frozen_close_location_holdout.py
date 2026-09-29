from __future__ import annotations

import pandas as pd

from src.entry_quality_holdout import _evaluate_frozen_entries, _trimmed_mean
from src.rolling_universe import RESEARCH_ANCHOR_UTC, run_rolling_universe_validation


RULE_NAME = "Frozen low-close-location v2.9"


def _baseline_events(details: pd.DataFrame) -> pd.DataFrame:
    if details is None or details.empty:
        return pd.DataFrame()

    base = details[
        details["Tier"].astype(str) == "Baseline LONG/BULL"
    ].copy()

    if base.empty:
        return base

    return (
        base.drop_duplicates(
            subset=["Period", "Symbol", "Event time"],
            keep="first",
        )
        .reset_index(drop=True)
    )


def freeze_close_location_rule(
    development_details: pd.DataFrame,
) -> dict:
    """Freeze the favorable lower quartile of 5m close location on development only."""
    base = _baseline_events(development_details)
    if base.empty:
        raise RuntimeError("No development baseline events are available for v2.9.")

    if "5m close location" not in base.columns:
        raise RuntimeError("5m close location is missing from development events.")

    values = pd.to_numeric(
        base["5m close location"],
        errors="coerce",
    ).dropna()

    if len(values) < 30:
        raise RuntimeError(
            f"Only {len(values)} complete development events are available; "
            "not enough to freeze the v2.9 quartile rule."
        )

    close_q25 = float(values.quantile(0.25))

    selected = base[
        pd.to_numeric(base["5m close location"], errors="coerce") <= close_q25
    ].copy()

    return {
        "name": RULE_NAME,
        "development_events": int(len(base)),
        "development_selected_events": int(len(selected)),
        "close_location_max": close_q25,
        "definition": "5m close location <= development Q25",
    }


def _apply_rule(details: pd.DataFrame, rule: dict) -> pd.DataFrame:
    base = _baseline_events(details)
    if base.empty:
        return base

    close = pd.to_numeric(base["5m close location"], errors="coerce")
    return base[close <= float(rule["close_location_max"])].copy()


def _top_symbol_share(frame: pd.DataFrame) -> float:
    if frame.empty or "Symbol" not in frame:
        return 0.0
    counts = frame["Symbol"].value_counts()
    return float(counts.iloc[0] / counts.sum() * 100.0) if not counts.empty else 0.0


def _summary_row(label: str, frame: pd.DataFrame, cost_bps: float) -> dict:
    if frame.empty:
        return {
            "Rule": label,
            "Events": 0,
            "Periods": 0,
            "Symbols": 0,
            "Top symbol share %": 0.0,
            "Gross avg bps": 0.0,
            "Gross median bps": 0.0,
            "Trimmed gross avg bps": 0.0,
            "Net avg bps": 0.0,
            "Net median bps": 0.0,
            "Net positive events %": 0.0,
            "Positive periods": "0/0",
            "Round-trip cost bps": float(cost_bps),
        }

    gross = pd.to_numeric(frame["Gross bps"], errors="coerce").dropna()
    net = pd.to_numeric(frame["Net bps"], errors="coerce").dropna()
    period_avg = frame.groupby("Period")["Net bps"].mean()

    return {
        "Rule": label,
        "Events": int(len(frame)),
        "Periods": int(frame["Period"].nunique()),
        "Symbols": int(frame["Symbol"].nunique()),
        "Top symbol share %": _top_symbol_share(frame),
        "Gross avg bps": float(gross.mean()) if not gross.empty else 0.0,
        "Gross median bps": float(gross.median()) if not gross.empty else 0.0,
        "Trimmed gross avg bps": _trimmed_mean(gross),
        "Net avg bps": float(net.mean()) if not net.empty else 0.0,
        "Net median bps": float(net.median()) if not net.empty else 0.0,
        "Net positive events %": float((net > 0).mean() * 100.0) if not net.empty else 0.0,
        "Positive periods": (
            f"{int((period_avg > 0).sum())}/{len(period_avg)}"
            if len(period_avg)
            else "0/0"
        ),
        "Round-trip cost bps": float(cost_bps),
    }


def _fraction(value: object) -> tuple[int, int]:
    try:
        left, right = str(value).split("/", 1)
        return int(left), int(right)
    except Exception:
        return 0, 0


def _verdict(summary: pd.DataFrame, rule: dict) -> tuple[str, str]:
    row = summary[summary["Rule"] == RULE_NAME]
    if row.empty:
        return "FAILED", "The frozen v2.9 rule produced no events in the fourth untouched window."

    row = row.iloc[0]
    events = int(row["Events"])
    periods = int(row["Periods"])
    symbols = int(row["Symbols"])
    top_share = float(row["Top symbol share %"])
    net_avg = float(row["Net avg bps"])
    net_median = float(row["Net median bps"])
    trimmed = float(row["Trimmed gross avg bps"])
    positive, total = _fraction(row["Positive periods"])
    cost = float(row["Round-trip cost bps"])

    enough_sample = events >= 15 and periods >= 6 and symbols >= 5
    not_concentrated = top_share <= 35.0
    robust_positive = (
        net_avg > 0
        and net_median > 0
        and trimmed > cost
        and total > 0
        and positive > total / 2
        and not_concentrated
    )

    threshold_text = (
        f"Frozen threshold: 5m close location <= "
        f"{float(rule['close_location_max']):.3f} (development Q25)."
    )
    metrics = (
        f"Fourth untouched result: {events} events/{periods} periods/{symbols} symbols, "
        f"net avg {net_avg:+.1f} bps, net median {net_median:+.1f} bps, "
        f"trimmed gross {trimmed:+.1f} bps, positive periods {positive}/{total}, "
        f"top-symbol share {top_share:.1f}%."
    )

    if enough_sample and robust_positive:
        return (
            "SURVIVED",
            threshold_text
            + " "
            + metrics
            + " The pre-frozen low-close-location rule survived the fourth untouched window. "
            "This is materially stronger out-of-sample evidence than the earlier candidates. "
            "Do not retune it; next step is forward paper validation on new data."
        )

    if not enough_sample:
        return (
            "LOW_SAMPLE",
            threshold_text
            + " "
            + metrics
            + " The fourth untouched sample is too small for a strong conclusion. "
            "Do not loosen the threshold after seeing this result."
        )

    if net_avg > 0 and not robust_positive:
        return (
            "MIXED",
            threshold_text
            + " "
            + metrics
            + " Average performance is positive, but median/period stability or symbol concentration "
            "does not support a validated edge. Do not retune on this untouched window."
        )

    return (
        "FAILED",
        threshold_text
        + " "
        + metrics
        + " The pre-frozen low-close-location rule did not survive the fourth untouched window. "
        "Reject this exact rule without retuning it on these results."
    )


def run_frozen_close_location_holdout(
    *,
    rule: dict,
    horizon_days: int = 120,
    end_offset_days: int = 510,
    lookback_days: int = 14,
    forward_days: int = 7,
    pool_size: int = 20,
    select_top_n: int = 5,
    min_daily_turnover_usd: float = 5_000_000.0,
    fee_bps: float = 4.0,
    slippage_bps: float = 2.0,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, dict, str, str]:
    rolling_periods, rolling_details, rolling_summary = run_rolling_universe_validation(
        horizon_days=int(horizon_days),
        lookback_days=int(lookback_days),
        forward_days=int(forward_days),
        pool_size=int(pool_size),
        select_top_n=int(select_top_n),
        min_daily_turnover_usd=float(min_daily_turnover_usd),
        end_offset_days=int(end_offset_days),
        as_of=RESEARCH_ANCHOR_UTC,
    )

    if rolling_details is None or rolling_details.empty:
        raise RuntimeError("Fourth untouched universe produced no selected symbols.")

    _, all_details = _evaluate_frozen_entries(
        rolling_details,
        forward_days=int(forward_days),
        fee_bps=float(fee_bps),
        slippage_bps=float(slippage_bps),
    )

    baseline = _baseline_events(all_details)
    frozen = _apply_rule(all_details, rule)
    cost_bps = 2.0 * (float(fee_bps) + float(slippage_bps))

    summary = pd.DataFrame(
        [
            _summary_row("Baseline LONG/BULL", baseline, cost_bps),
            _summary_row(RULE_NAME, frozen, cost_bps),
        ]
    )

    status, verdict = _verdict(summary, rule)

    if not frozen.empty:
        frozen = frozen.copy()
        frozen["Frozen rule"] = RULE_NAME

    return summary, frozen, rolling_periods, rolling_summary, status, verdict
