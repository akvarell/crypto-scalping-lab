from __future__ import annotations

import pandas as pd

from src.alt_basket_study import _alt_context, _cooldown, _events, _prepare_coin
from src.scalping_edge_map import (
    _add_1m_micro_features,
    _entry_index_confirmed,
    _fetch_1m,
    _resample_5m,
)


STOP_BPS = [20, 30, 40, 60]
TAKE_BPS = [20, 30, 40, 60, 80]
MAX_HOLD_MINUTES = 10


def _trimmed_mean(series: pd.Series) -> float:
    clean = series.dropna().astype(float)
    if clean.empty:
        return 0.0
    if len(clean) < 20:
        return float(clean.mean())
    lo = clean.quantile(0.10)
    hi = clean.quantile(0.90)
    trimmed = clean[(clean >= lo) & (clean <= hi)]
    return float(trimmed.mean()) if not trimmed.empty else float(clean.mean())


def _simulate_long_exit(
    one_minute: pd.DataFrame,
    entry_i: int,
    *,
    stop_bps: int | None,
    take_bps: int | None,
    max_hold_minutes: int = MAX_HOLD_MINUTES,
) -> dict | None:
    if entry_i >= len(one_minute):
        return None

    entry = float(one_minute.iloc[entry_i]["open"])
    if entry <= 0:
        return None

    last_i = min(entry_i + int(max_hold_minutes) - 1, len(one_minute) - 1)
    if last_i < entry_i:
        return None

    stop_price = None if stop_bps is None else entry * (1.0 - float(stop_bps) / 10_000.0)
    take_price = None if take_bps is None else entry * (1.0 + float(take_bps) / 10_000.0)

    for i in range(entry_i, last_i + 1):
        row = one_minute.iloc[i]
        low = float(row["low"])
        high = float(row["high"])

        stop_hit = stop_price is not None and low <= stop_price
        take_hit = take_price is not None and high >= take_price

        # Conservative same-minute assumption: stop wins if both are touched.
        if stop_hit:
            return {
                "gross_bps": -float(stop_bps),
                "exit_reason": "STOP",
                "exit_time": one_minute.index[i],
                "hold_minutes": i - entry_i + 1,
            }
        if take_hit:
            return {
                "gross_bps": float(take_bps),
                "exit_reason": "TAKE",
                "exit_time": one_minute.index[i],
                "hold_minutes": i - entry_i + 1,
            }

    exit_price = float(one_minute.iloc[last_i]["close"])
    return {
        "gross_bps": (exit_price / entry - 1.0) * 10_000.0,
        "exit_reason": "TIME",
        "exit_time": one_minute.index[last_i],
        "hold_minutes": last_i - entry_i + 1,
    }


def run_exit_surface(
    rolling_details: pd.DataFrame,
    *,
    forward_days: int,
    fee_bps: float,
    slippage_bps: float,
    max_periods: int = 12,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Coarse exit study for the frozen v2.0 development candidate.

    Candidate is fixed before this study:
      - 5m alt-basket event
      - LONG
      - alt BULL regime
      - 1m volume-confirm entry
      - max hold 10 minutes

    Only the coarse stop/take exit grid is varied here.
    """
    if rolling_details is None or rolling_details.empty:
        raise ValueError("Run Rolling Universe Validation first.")

    selected = rolling_details[["Period", "Selection time", "Symbol"]].drop_duplicates().copy()
    selected["Selection time"] = pd.to_datetime(selected["Selection time"], utc=True)
    keep_periods = sorted(selected["Period"].unique())[-int(max_periods):]
    selected = selected[selected["Period"].isin(keep_periods)]

    cost_bps = 2.0 * (float(fee_bps) + float(slippage_bps))
    event_entries = []

    for period, period_sel in selected.groupby("Period", sort=True):
        selection_time = pd.Timestamp(period_sel["Selection time"].iloc[0])
        end = selection_time + pd.Timedelta(days=int(forward_days))

        one_minute_data: dict[str, pd.DataFrame] = {}
        five_minute_prepared: dict[str, pd.DataFrame] = {}

        for symbol in period_sel["Symbol"].astype(str).tolist():
            try:
                one_minute = _fetch_1m(symbol, selection_time, end)
            except Exception:
                one_minute = pd.DataFrame()

            if one_minute.empty or len(one_minute) < 1000:
                continue

            one_minute = _add_1m_micro_features(one_minute)
            five_minute = _resample_5m(one_minute)
            if len(five_minute) < 200:
                continue

            one_minute_data[symbol] = one_minute
            five_minute_prepared[symbol] = _prepare_coin(five_minute)

        if len(five_minute_prepared) < 3:
            continue

        contexts = _alt_context(five_minute_prepared)

        for symbol, five_minute in five_minute_prepared.items():
            one_minute = one_minute_data[symbol]
            ctx = contexts.get(symbol)
            if ctx is None:
                continue

            events = _cooldown(_events(five_minute, ctx), minutes=60)
            events = events[
                (events["direction"] == 1)
                & (events["alt_regime"] == "BULL")
            ]
            if events.empty:
                continue

            for event_time, event in events.iterrows():
                known_time = pd.Timestamp(event_time) + pd.Timedelta(minutes=5)
                entry_i = _entry_index_confirmed(
                    one_minute,
                    known_time,
                    1,
                    confirm_window_minutes=3,
                )
                if entry_i is None:
                    continue

                event_entries.append(
                    {
                        "Period": int(period),
                        "Selection time": selection_time,
                        "Symbol": symbol,
                        "Event time": event_time,
                        "Entry index": int(entry_i),
                        "One minute": one_minute,
                    }
                )

    if not event_entries:
        raise RuntimeError("No LONG/BULL 1m-confirm events were available.")

    rows = []

    # Baseline fixed 10-minute exit.
    configs = [("Fixed 10m", None, None)]
    configs += [
        (f"SL {stop} / TP {take}", stop, take)
        for stop in STOP_BPS
        for take in TAKE_BPS
    ]

    for item in event_entries:
        for label, stop_bps, take_bps in configs:
            result = _simulate_long_exit(
                item["One minute"],
                item["Entry index"],
                stop_bps=stop_bps,
                take_bps=take_bps,
                max_hold_minutes=MAX_HOLD_MINUTES,
            )
            if result is None:
                continue

            rows.append(
                {
                    "Config": label,
                    "Stop bps": stop_bps,
                    "Take bps": take_bps,
                    "Period": item["Period"],
                    "Selection time": item["Selection time"],
                    "Symbol": item["Symbol"],
                    "Event time": item["Event time"],
                    "Gross bps": float(result["gross_bps"]),
                    "Net bps": float(result["gross_bps"]) - cost_bps,
                    "Exit reason": result["exit_reason"],
                    "Hold minutes": int(result["hold_minutes"]),
                }
            )

    details = pd.DataFrame(rows)
    if details.empty:
        raise RuntimeError("Exit surface produced no completed events.")

    summary_rows = []
    for config, group in details.groupby("Config", sort=False):
        gross = group["Gross bps"].astype(float)
        net = group["Net bps"].astype(float)
        period_avg = group.groupby("Period")["Net bps"].mean()

        reasons = group["Exit reason"].value_counts(normalize=True) * 100.0

        summary_rows.append(
            {
                "Config": config,
                "Events": int(len(group)),
                "Periods": int(group["Period"].nunique()),
                "Gross avg bps": float(gross.mean()),
                "Gross median bps": float(gross.median()),
                "Trimmed gross avg bps": _trimmed_mean(gross),
                "Net avg bps": float(net.mean()),
                "Net median bps": float(net.median()),
                "Net positive events %": float((net > 0).mean() * 100.0),
                "Positive period avg": (
                    f"{int((period_avg > 0).sum())}/{len(period_avg)}"
                    if len(period_avg)
                    else "0/0"
                ),
                "Worst period avg bps": float(period_avg.min()) if len(period_avg) else 0.0,
                "Best period avg bps": float(period_avg.max()) if len(period_avg) else 0.0,
                "Take exits %": float(reasons.get("TAKE", 0.0)),
                "Stop exits %": float(reasons.get("STOP", 0.0)),
                "Time exits %": float(reasons.get("TIME", 0.0)),
                "Avg hold min": float(group["Hold minutes"].mean()),
                "Round-trip cost bps": cost_bps,
            }
        )

    summary = pd.DataFrame(summary_rows)
    return summary, details
