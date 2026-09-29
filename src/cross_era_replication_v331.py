from __future__ import annotations

import numpy as np
import pandas as pd

from src.dynamic_scalping import _fetch_5m
from src.event_dataset_v3 import _forward_return_bps
from src.microstructure_context_v33 import (
    _cluster_spread_ci,
    _context_frame,
    _prepare_extended,
    _quartile_stats,
    _spearman,
)
from src.scalping_edge_map import _fetch_1m


FIXED_HORIZON_MIN = 15
SAMPLE_MINUTES = 60


def build_relative_z_short_period_v331(
    period_frame: pd.DataFrame,
    *,
    forward_days: int = 7,
    warmup_hours: int = 48,
    sample_minutes: int = SAMPLE_MINUTES,
) -> pd.DataFrame:
    """Replicate the exact v3.3 descriptive mechanism on an older era.

    Frozen mechanism:
    - relative_z > 0
    - SHORT reversion
    - 15m horizon
    - hourly deterministic sampling
    - immediate next-tradable 1m execution after known_time

    No magnitude threshold is optimized.
    """
    if period_frame is None or period_frame.empty:
        return pd.DataFrame()

    period = int(period_frame["Period"].iloc[0])
    selection_time = pd.Timestamp(period_frame["Selection time"].iloc[0])
    period_end = selection_time + pd.Timedelta(days=int(forward_days))
    fetch_start = selection_time - pd.Timedelta(hours=int(warmup_hours))

    context_symbols = (
        period_frame[period_frame["Context selected"]]["Symbol"]
        .astype(str)
        .tolist()
    )
    trade_symbols = (
        period_frame[period_frame["Trade selected"]]["Symbol"]
        .astype(str)
        .tolist()
    )

    prepared: dict[str, pd.DataFrame] = {}
    for symbol in context_symbols:
        try:
            raw = _fetch_5m(symbol, fetch_start, period_end)
        except Exception:
            raw = pd.DataFrame()

        if raw.empty or len(raw) < 600:
            continue

        prepared[symbol] = _prepare_extended(raw)

    if len(prepared) < 8:
        return pd.DataFrame()

    rows: list[dict] = []

    for symbol in trade_symbols:
        coin = prepared.get(symbol)
        if coin is None:
            continue

        ctx = _context_frame(symbol, prepared)
        if ctx.empty:
            continue

        aligned = coin.join(ctx, how="left")
        aligned["relative_ret_15m"] = (
            aligned["ret_15m"] - aligned["basket_ret_15m"]
        )
        aligned["relative_z"] = (
            aligned["relative_ret_15m"]
            / aligned["dispersion_15m"].replace(0.0, float("nan"))
        )

        sample_index = aligned.index[
            (aligned.index >= selection_time)
            & (aligned.index < period_end)
            & (aligned.index.minute % int(sample_minutes) == 0)
        ]

        if len(sample_index) == 0:
            continue

        try:
            minute = _fetch_1m(
                symbol,
                selection_time - pd.Timedelta(minutes=5),
                period_end + pd.Timedelta(minutes=25),
            )
        except Exception:
            minute = pd.DataFrame()

        if minute.empty:
            continue

        for ts in sample_index:
            relative_z = float(
                aligned.loc[ts].get("relative_z", float("nan"))
            )
            if not np.isfinite(relative_z) or relative_z <= 0:
                continue

            known_time = pd.Timestamp(ts) + pd.Timedelta(minutes=5)
            entry_i = int(
                minute.index.searchsorted(known_time, side="left")
            )
            if entry_i >= len(minute):
                continue

            gross_short = _forward_return_bps(
                minute,
                entry_i,
                -1,
                FIXED_HORIZON_MIN,
            )
            if not np.isfinite(gross_short):
                continue

            rows.append(
                {
                    "Period": int(period),
                    "Selection time": selection_time,
                    "Symbol": symbol,
                    "Event time": ts,
                    "Known time": known_time,
                    "Entry time": minute.index[entry_i],
                    "Feature": "Relative-z reversion",
                    "Side": "SHORT",
                    "Strength": relative_z,
                    "Gross 15m bps": float(gross_short),
                }
            )

    return pd.DataFrame(rows)


def _fraction(value: object) -> float:
    try:
        left, right = str(value).split("/", 1)
        total = int(right)
        return int(left) / total if total else 0.0
    except Exception:
        return 0.0


def _summarize_replication_sample(
    frame: pd.DataFrame,
    *,
    sample_label: str,
) -> dict:
    work = frame[
        ["Period", "Symbol", "Strength", "Gross 15m bps"]
    ].copy()
    work = work.dropna()

    if work.empty:
        return {
            "Sample": sample_label,
            "Events": 0,
            "Periods": 0,
            "Symbols": 0,
            "Spearman": float("nan"),
            "Q4-Q1 gross spread bps": float("nan"),
            "Q4 gross avg bps": float("nan"),
            "Q4 gross median bps": float("nan"),
            "Q4 trimmed gross bps": float("nan"),
            "Q4 positive periods": "0/0",
            "Q4 top symbol share %": float("nan"),
            "99% cluster spread CI low": float("nan"),
            "99% cluster spread CI high": float("nan"),
        }

    rho = _spearman(
        work["Strength"],
        work["Gross 15m bps"],
        min_n=100,
    )
    q = _quartile_stats(work, "Gross 15m bps")
    ci_low, ci_high = _cluster_spread_ci(
        work,
        "Gross 15m bps",
        repeats=500,
        seed=46,
    )

    return {
        "Sample": sample_label,
        "Events": int(len(work)),
        "Periods": int(work["Period"].nunique()),
        "Symbols": int(work["Symbol"].nunique()),
        "Spearman": rho,
        "Q4-Q1 gross spread bps": (
            q["spread"] if q else float("nan")
        ),
        "Q4 gross avg bps": (
            q["q4_avg"] if q else float("nan")
        ),
        "Q4 gross median bps": (
            q["q4_median"] if q else float("nan")
        ),
        "Q4 trimmed gross bps": (
            q["q4_trimmed"] if q else float("nan")
        ),
        "Q4 positive periods": (
            q["q4_positive_periods"] if q else "0/0"
        ),
        "Q4 top symbol share %": (
            q["q4_top_symbol_share"] if q else float("nan")
        ),
        "99% cluster spread CI low": ci_low,
        "99% cluster spread CI high": ci_high,
    }


def analyze_cross_era_replication_v331(
    observations: pd.DataFrame,
) -> tuple[pd.DataFrame, str, str]:
    """Evaluate the frozen descriptive mechanism on two inspected older eras."""
    if observations is None or observations.empty:
        raise ValueError("v3.3.1 replication observations are empty.")

    if "Era" not in observations.columns:
        raise ValueError("v3.3.1 observations require an Era column.")

    rows = []
    for era in ["Era A", "Era B"]:
        era_frame = observations[
            observations["Era"].astype(str) == era
        ].copy()
        rows.append(
            _summarize_replication_sample(
                era_frame,
                sample_label=era,
            )
        )

    pooled = observations.copy()
    pooled["Period"] = (
        pooled["Era"].astype(str)
        + "-"
        + pooled["Period"].astype(str)
    )
    rows.append(
        _summarize_replication_sample(
            pooled,
            sample_label="Pooled older eras",
        )
    )

    summary = pd.DataFrame(rows)

    era_a = summary[summary["Sample"] == "Era A"].iloc[0]
    era_b = summary[summary["Sample"] == "Era B"].iloc[0]
    pooled_row = summary[
        summary["Sample"] == "Pooled older eras"
    ].iloc[0]

    def era_pass(row: pd.Series) -> bool:
        return (
            int(row["Events"]) >= 200
            and int(row["Periods"]) >= 8
            and int(row["Symbols"]) >= 8
            and np.isfinite(float(row["Spearman"]))
            and float(row["Spearman"]) > 0
            and np.isfinite(float(row["Q4-Q1 gross spread bps"]))
            and float(row["Q4-Q1 gross spread bps"]) > 0
            and np.isfinite(float(row["Q4 gross median bps"]))
            and float(row["Q4 gross median bps"]) > 0
            and _fraction(row["Q4 positive periods"]) > 0.5
        )

    pooled_pass = (
        int(pooled_row["Events"]) >= 500
        and int(pooled_row["Periods"]) >= 16
        and int(pooled_row["Symbols"]) >= 8
        and np.isfinite(float(pooled_row["Q4 gross median bps"]))
        and float(pooled_row["Q4 gross median bps"]) > 0
        and np.isfinite(float(pooled_row["99% cluster spread CI low"]))
        and float(pooled_row["99% cluster spread CI low"]) > 0
        and np.isfinite(float(pooled_row["Q4 top symbol share %"]))
        and float(pooled_row["Q4 top symbol share %"]) <= 35.0
    )

    replicated = era_pass(era_a) and era_pass(era_b) and pooled_pass
    status = "REPLICATED" if replicated else "NOT_REPLICATED"

    def fmt(row: pd.Series) -> str:
        return (
            f"{row['Sample']}: {int(row['Events'])} events/"
            f"{int(row['Periods'])} periods/{int(row['Symbols'])} symbols, "
            f"rho {float(row['Spearman']):+.2f}, "
            f"Q4-Q1 {float(row['Q4-Q1 gross spread bps']):+.1f} bps, "
            f"Q4 median {float(row['Q4 gross median bps']):+.1f} bps, "
            f"positive periods {row['Q4 positive periods']}, "
            f"99% CI low {float(row['99% cluster spread CI low']):+.1f} bps."
        )

    if replicated:
        verdict = (
            "The v3.3 Relative-z reversion / SHORT / 15m mechanism replicated across both "
            "older inspected eras without relaxing the original v3.3 gate. "
            + fmt(era_a)
            + " "
            + fmt(era_b)
            + " "
            + fmt(pooled_row)
            + " This is cross-era development evidence, not strategy validation. "
            "The mechanism may now advance to purged chronological walk-forward, where any "
            "trade threshold must be learned inside each training fold only."
        )
    else:
        verdict = (
            "The v3.3 near-miss mechanism did not replicate cleanly across both older eras. "
            + fmt(era_a)
            + " "
            + fmt(era_b)
            + " "
            + fmt(pooled_row)
            + " Do not rescue it by changing the side, horizon or thresholds. "
            "Return to genuinely different mechanism/data-source research."
        )

    return summary, status, verdict
