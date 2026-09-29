from __future__ import annotations

import numpy as np
import pandas as pd

from src.dynamic_scalping import _fetch_5m
from src.event_dataset_v3 import (
    _context_for_trade_symbol,
    _forward_return_bps,
    _prepare_5m_v3,
)
from src.event_family_benchmark_v31 import (
    FAMILIES,
    _family_events,
)
from src.scalping_edge_map import _fetch_1m


HORIZONS = [1, 3, 5, 10, 15, 30]
EXECUTIONS = ["Immediate", "+1m delay"]


def _merge_windows_v32(
    event_times: list[pd.Timestamp],
) -> list[dict]:
    """Merge 1m windows while keeping enough future data for 30m exits."""
    windows: list[dict] = []

    for event_time in sorted(pd.Timestamp(ts) for ts in event_times):
        known_time = event_time + pd.Timedelta(minutes=5)
        start = known_time - pd.Timedelta(minutes=5)
        end = known_time + pd.Timedelta(minutes=35)

        if not windows or start > windows[-1]["end"]:
            windows.append(
                {
                    "start": start,
                    "end": end,
                    "events": [event_time],
                }
            )
        else:
            windows[-1]["end"] = max(windows[-1]["end"], end)
            windows[-1]["events"].append(event_time)

    return windows


def build_execution_horizon_period_v32(
    period_frame: pd.DataFrame,
    *,
    forward_days: int = 7,
    warmup_hours: int = 48,
) -> pd.DataFrame:
    """Build Immediate and +1m-delay outcomes for all v3.1 event families."""
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
    trade_symbols = set(
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

        prepared[symbol] = _prepare_5m_v3(raw)

    if len(prepared) < 8:
        return pd.DataFrame()

    output_rows: list[dict] = []

    for symbol in sorted(trade_symbols):
        coin = prepared.get(symbol)
        if coin is None:
            continue

        ctx = _context_for_trade_symbol(symbol, prepared)
        if ctx.empty:
            continue

        events = _family_events(
            symbol,
            coin,
            ctx,
            prepared,
            selection_time=selection_time,
            period_end=period_end,
        )
        if events.empty:
            continue

        event_lookup: dict[pd.Timestamp, list[pd.Series]] = {}
        for ts, row in events.iterrows():
            event_lookup.setdefault(pd.Timestamp(ts), []).append(row)

        for batch in _merge_windows_v32(list(event_lookup)):
            try:
                minute_batch = _fetch_1m(
                    symbol,
                    batch["start"],
                    batch["end"],
                )
            except Exception:
                minute_batch = pd.DataFrame()

            if minute_batch.empty:
                continue

            for event_time in batch["events"]:
                known_time = pd.Timestamp(event_time) + pd.Timedelta(minutes=5)
                local_start = known_time
                local_end = known_time + pd.Timedelta(minutes=35)
                one_minute = minute_batch[
                    (minute_batch.index >= local_start)
                    & (minute_batch.index < local_end)
                ].copy()

                if one_minute.empty or len(one_minute) < 31:
                    continue

                immediate_i = int(
                    one_minute.index.searchsorted(known_time, side="left")
                )
                if immediate_i >= len(one_minute):
                    continue

                execution_indices = {
                    "Immediate": immediate_i,
                    "+1m delay": immediate_i + 1,
                }

                for event in event_lookup[event_time]:
                    direction = int(event["Direction"])
                    side = "LONG" if direction == 1 else "SHORT"

                    for execution, entry_i in execution_indices.items():
                        if entry_i >= len(one_minute):
                            continue

                        record = {
                            "Period": period,
                            "Symbol": symbol,
                            "Family": str(event["Family"]),
                            "Side": side,
                            "Execution": execution,
                            "Event time": event_time,
                            "Known time": known_time,
                            "Entry time": one_minute.index[entry_i],
                        }

                        for horizon in HORIZONS:
                            record[f"Gross {horizon}m bps"] = _forward_return_bps(
                                one_minute,
                                entry_i,
                                direction,
                                horizon,
                            )

                        output_rows.append(record)

    return pd.DataFrame(output_rows)


def _trimmed_mean(series: pd.Series) -> float:
    clean = pd.to_numeric(series, errors="coerce").dropna()
    if clean.empty:
        return float("nan")
    if len(clean) < 20:
        return float(clean.mean())
    lo = clean.quantile(0.10)
    hi = clean.quantile(0.90)
    trimmed = clean[(clean >= lo) & (clean <= hi)]
    return float(trimmed.mean()) if not trimmed.empty else float(clean.mean())


def _period_bootstrap_net_ci(
    frame: pd.DataFrame,
    *,
    gross_col: str,
    cost_bps: float,
    repeats: int = 500,
    seed: int = 44,
) -> tuple[float, float]:
    periods = sorted(frame["Period"].dropna().unique().tolist())
    if len(periods) < 6:
        return float("nan"), float("nan")

    rng = np.random.default_rng(seed)
    values: list[float] = []

    for _ in range(int(repeats)):
        sampled = rng.choice(periods, size=len(periods), replace=True)
        parts = []
        for period in sampled:
            gross = pd.to_numeric(
                frame.loc[frame["Period"] == period, gross_col],
                errors="coerce",
            ).dropna()
            if not gross.empty:
                parts.append(gross)

        if not parts:
            continue

        sample = pd.concat(parts, ignore_index=True)
        values.append(float((sample - float(cost_bps)).mean()))

    if len(values) < 100:
        return float("nan"), float("nan")

    return (
        float(np.quantile(values, 0.005)),
        float(np.quantile(values, 0.995)),
    )


def _positive_period_fraction(value: object) -> float:
    try:
        left, right = str(value).split("/", 1)
        total = int(right)
        return int(left) / total if total else 0.0
    except Exception:
        return 0.0


def benchmark_execution_horizon_v32(
    events: pd.DataFrame,
    *,
    costs_bps: tuple[float, ...] = (12.0, 20.0, 30.0),
) -> tuple[pd.DataFrame, pd.DataFrame, str]:
    """Map execution delay and holding horizon without tuning event thresholds."""
    if events is None or events.empty:
        raise ValueError("v3.2 execution/horizon dataset is empty.")

    rows: list[dict] = []

    for family in FAMILIES:
        family_frame = events[events["Family"].astype(str) == family]
        if family_frame.empty:
            continue

        for side in ["ALL", "LONG", "SHORT"]:
            side_frame = (
                family_frame
                if side == "ALL"
                else family_frame[family_frame["Side"].astype(str) == side]
            )
            if side_frame.empty:
                continue

            for execution in EXECUTIONS:
                exec_frame = side_frame[
                    side_frame["Execution"].astype(str) == execution
                ]
                if exec_frame.empty:
                    continue

                for horizon in HORIZONS:
                    gross_col = f"Gross {horizon}m bps"
                    gross = pd.to_numeric(
                        exec_frame[gross_col],
                        errors="coerce",
                    )
                    valid = exec_frame.loc[gross.notna()].copy()
                    valid[gross_col] = gross.dropna().values

                    if valid.empty:
                        continue

                    top_symbol_share = float(
                        valid["Symbol"].value_counts().iloc[0]
                        / len(valid)
                        * 100.0
                    )

                    for cost in costs_bps:
                        net = valid[gross_col] - float(cost)
                        period_avg = (
                            valid.assign(_net=net)
                            .groupby("Period")["_net"]
                            .mean()
                        )
                        positive_periods = (
                            f"{int((period_avg > 0).sum())}/{len(period_avg)}"
                            if len(period_avg)
                            else "0/0"
                        )
                        ci_low, ci_high = _period_bootstrap_net_ci(
                            valid,
                            gross_col=gross_col,
                            cost_bps=float(cost),
                        )

                        strict_candidate = (
                            float(cost) == 12.0
                            and len(valid) >= 60
                            and valid["Period"].nunique() >= 8
                            and valid["Symbol"].nunique() >= 8
                            and float(net.mean()) > 0
                            and float(net.median()) > 0
                            and _trimmed_mean(valid[gross_col]) > float(cost)
                            and _positive_period_fraction(positive_periods) > 0.5
                            and top_symbol_share <= 35.0
                            and np.isfinite(ci_low)
                            and ci_low > 0
                        )

                        rows.append(
                            {
                                "Family": family,
                                "Side": side,
                                "Execution": execution,
                                "Horizon min": int(horizon),
                                "Round-trip cost bps": float(cost),
                                "Events": int(len(valid)),
                                "Periods": int(valid["Period"].nunique()),
                                "Symbols": int(valid["Symbol"].nunique()),
                                "Top symbol share %": top_symbol_share,
                                "Gross avg bps": float(valid[gross_col].mean()),
                                "Gross median bps": float(valid[gross_col].median()),
                                "Trimmed gross avg bps": _trimmed_mean(
                                    valid[gross_col]
                                ),
                                "Net avg bps": float(net.mean()),
                                "Net median bps": float(net.median()),
                                "Positive periods": positive_periods,
                                "99% cluster net CI low": ci_low,
                                "99% cluster net CI high": ci_high,
                                "Strict candidate": (
                                    "YES" if strict_candidate else "NO"
                                ),
                            }
                        )

    surface = pd.DataFrame(rows)
    if surface.empty:
        raise RuntimeError("v3.2 produced no comparable execution/horizon results.")

    base12 = surface[
        surface["Round-trip cost bps"] == 12.0
    ].copy()

    # Neighbor support reduces the chance that one isolated horizon wins by luck.
    candidate_rows = []
    for idx, row in base12.iterrows():
        if row["Strict candidate"] != "YES":
            continue

        same = base12[
            (base12["Family"] == row["Family"])
            & (base12["Side"] == row["Side"])
            & (base12["Execution"] == row["Execution"])
        ].copy()

        h = int(row["Horizon min"])
        neighbors = {
            1: [3],
            3: [1, 5],
            5: [3, 10],
            10: [5, 15],
            15: [10, 30],
            30: [15],
        }[h]

        supported = same[
            same["Horizon min"].isin(neighbors)
            & (same["Net avg bps"] > 0)
            & (same["Net median bps"] > 0)
            & (same["99% cluster net CI low"] > 0)
        ]

        enriched = row.to_dict()
        enriched["Neighbor support"] = "YES" if not supported.empty else "NO"
        candidate_rows.append(enriched)

    candidates = pd.DataFrame(candidate_rows)

    if candidates.empty:
        best = base12.sort_values(
            ["Net avg bps", "Net median bps"],
            ascending=False,
        ).iloc[0]
        verdict = (
            "No execution/horizon combination passed the strict 12 bps screen. "
            f"The strongest descriptive combination was {best['Family']} / {best['Side']} / "
            f"{best['Execution']} / {int(best['Horizon min'])}m with net avg "
            f"{float(best['Net avg bps']):+.1f} bps, net median "
            f"{float(best['Net median bps']):+.1f} bps, positive periods "
            f"{best['Positive periods']} and 99% CI low "
            f"{float(best['99% cluster net CI low']):+.1f} bps. "
            "This is diagnostic only."
        )
    else:
        robust = candidates[
            candidates["Neighbor support"].astype(str) == "YES"
        ]
        shown = robust if not robust.empty else candidates

        descriptions = []
        for _, row in shown.sort_values(
            "Net avg bps",
            ascending=False,
        ).head(3).iterrows():
            descriptions.append(
                f"{row['Family']} / {row['Side']} / {row['Execution']} / "
                f"{int(row['Horizon min'])}m "
                f"(net {float(row['Net avg bps']):+.1f} bps, median "
                f"{float(row['Net median bps']):+.1f}, periods "
                f"{row['Positive periods']}, 99% CI low "
                f"{float(row['99% cluster net CI low']):+.1f}, neighbor "
                f"{row['Neighbor support']})"
            )

        verdict = (
            "v3.2 found strict timing candidates after 12 bps costs: "
            + "; ".join(descriptions)
            + ". Because many timing combinations were inspected, this is still development "
            "evidence only; the next step must be purged walk-forward, not another fixed holdout."
        )

    return surface, candidates, verdict
