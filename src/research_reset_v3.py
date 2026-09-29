from __future__ import annotations

import pandas as pd

from src.event_dataset_v3 import build_event_dataset_v3
from src.research_universe_v3 import build_research_universe_v3
from src.rolling_universe import RESEARCH_ANCHOR_UTC
from src.validation_v3 import (
    audit_event_dataset_v3,
    cost_stress_v3,
    feature_stability_v3,
)


def _summary_row(step: int, test: str, status: str, message: str) -> dict:
    return {
        "Step": int(step),
        "Test": test,
        "Status": status,
        "What became clear": message,
    }


def run_research_reset_v3(
    *,
    horizon_days: int = 90,
    lookback_days: int = 14,
    forward_days: int = 7,
    pool_size: int = 20,
    trade_top_n: int = 5,
    context_top_n: int = 15,
    min_daily_turnover_usd: float = 5_000_000.0,
    progress_callback=None,
    summary_callback=None,
) -> dict:
    total = 5
    summaries = []

    def progress(step: int, label: str) -> None:
        if progress_callback is not None:
            progress_callback(step, total, label)

    def emit(step: int, label: str, status: str, message: str) -> None:
        summaries.append(_summary_row(step, label, status, message))
        if summary_callback is not None:
            summary_callback(step, total, label, status, message)

    progress(1, "v3 Historical Universe")
    periods, universe_details, universe_summary = build_research_universe_v3(
        horizon_days=int(horizon_days),
        lookback_days=int(lookback_days),
        forward_days=int(forward_days),
        pool_size=int(pool_size),
        trade_top_n=int(trade_top_n),
        context_top_n=int(context_top_n),
        min_daily_turnover_usd=float(min_daily_turnover_usd),
        end_offset_days=0,
        as_of=RESEARCH_ANCHOR_UTC,
    )

    universe_status = "PASS_WITH_CAVEAT" if len(periods) >= 8 else "REVIEW"
    universe_message = (
        f"Built {len(periods)} reproducible periods with top-{trade_top_n} trade symbols and "
        f"top-{context_top_n} context symbols. Trade and regime universes are now separated. "
        f"Survivorship status: {universe_summary['survivorship_status']}. "
        "Delisted historical symbols are still not reconstructed, so old screener uplift must "
        "not be treated as fully survivorship-free."
    )
    emit(1, "v3 Historical Universe", universe_status, universe_message)

    progress(2, "Causal Event Dataset")
    events = build_event_dataset_v3(
        universe_details,
        forward_days=int(forward_days),
        warmup_hours=48,
    )

    if events.empty:
        raise RuntimeError("v3 causal event dataset returned no events.")

    event_status = (
        "PASS"
        if len(events) >= 100
        and events["Period"].nunique() >= 8
        and events["Symbol"].nunique() >= 8
        else "REVIEW"
    )
    event_message = (
        f"Built {len(events)} causal breakout events across {events['Period'].nunique()} periods "
        f"and {events['Symbol'].nunique()} symbols. Every 5m calculation now has 48h warm-up; "
        "relative volume uses prior candles only; volume/order-flow/candle-shape are features, "
        "not pre-selection thresholds; execution waits for one completed 1m decision candle "
        "and enters on the following 1m open."
    )
    emit(2, "Causal Event Dataset", event_status, event_message)

    progress(3, "Timing & Data Audit")
    audit, audit_verdict = audit_event_dataset_v3(
        universe_details,
        events,
    )
    hard_fail = bool((audit["Status"].astype(str) == "FAIL").any())
    audit_status = "FAILED" if hard_fail else "PASS"
    emit(3, "Timing & Data Audit", audit_status, audit_verdict)

    progress(4, "Cost Stress")
    cost_table = cost_stress_v3(events)
    base12 = cost_table[
        (cost_table["Side"] == "ALL")
        & (cost_table["Round-trip cost bps"] == 12.0)
    ]
    if base12.empty:
        cost_status = "REVIEW"
        cost_message = "Could not calculate the 12 bps broad-event cost baseline."
    else:
        row = base12.iloc[0]
        net = float(row["Net avg bps"])
        median = float(row["Net median bps"])
        cost_status = "BASELINE_POSITIVE" if net > 0 and median > 0 else "BASELINE_NEGATIVE"
        cost_message = (
            f"Broad, unfiltered breakout events at 12 bps round-trip cost: "
            f"net avg {net:+.1f} bps, net median {median:+.1f} bps, "
            f"positive periods {row['Positive periods']}. "
            "This is a reference baseline, not a strategy. The same table also stress-tests "
            "20 and 30 bps costs and LONG/SHORT separately."
        )
    emit(4, "Cost Stress", cost_status, cost_message)

    progress(5, "Period-Cluster Feature Stability")
    feature_table, feature_verdict = feature_stability_v3(events)
    if feature_table.empty:
        feature_status = "NO_CANDIDATE"
    else:
        candidate_count = int(
            (feature_table["Candidate feature"].astype(str) == "YES").sum()
        )
        feature_status = "CANDIDATE_FOUND" if candidate_count > 0 else "NO_CANDIDATE"
    emit(5, "Period-Cluster Feature Stability", feature_status, feature_verdict)

    if hard_fail:
        next_step = (
            "Stop. Fix the v3 timing/data audit failure before any strategy research."
        )
    elif feature_status == "CANDIDATE_FOUND":
        next_step = (
            "Do not open another historical holdout. Next build v3.1 Purged Walk-Forward: "
            "freeze feature direction inside each training block, derive thresholds only from "
            "past training periods, purge the boundary, and evaluate the next chronological block "
            "with 12/20/30 bps cost stress plus portfolio concurrency limits."
        )
    else:
        next_step = (
            "Do not open another historical holdout. No feature is stable enough yet. "
            "Next add a genuinely different development-only feature family or simplify the "
            "event definition, then rerun the same v3 stability screen."
        )

    return {
        "periods": periods,
        "universe_details": universe_details,
        "universe_summary": universe_summary,
        "events": events,
        "audit": audit,
        "cost_table": cost_table,
        "feature_table": feature_table,
        "stage_summaries": pd.DataFrame(summaries),
        "next_step": next_step,
        "research_anchor": RESEARCH_ANCHOR_UTC.isoformat(),
    }
