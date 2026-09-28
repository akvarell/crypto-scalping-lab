from __future__ import annotations

import pandas as pd

from src.alt_basket_study import _alt_context, _cooldown, _events, _prepare_coin
from src.scalping_edge_map import (
    _add_1m_micro_features,
    _entry_index_confirmed,
    _fetch_1m,
    _resample_5m,
)


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


def _tiers(event: pd.Series, confirm_row: pd.Series) -> list[str]:
    """Fixed coarse quality buckets using only information known before entry."""
    labels = ["Baseline LONG/BULL"]

    confirm_vol = float(confirm_row.get("volume_ratio_1m", 0.0))
    confirm_open = float(confirm_row.get("open", 0.0))
    confirm_close = float(confirm_row.get("close", 0.0))
    confirm_body_bps = (
        (confirm_close / confirm_open - 1.0) * 10_000.0
        if confirm_open > 0
        else 0.0
    )

    event_vol = float(event.get("coin_volume_ratio", 0.0))
    event_range = float(event.get("range_atr", 0.0))
    event_relative = float(event.get("relative_move_atr", 0.0))

    if confirm_vol >= 2.5 and confirm_body_bps >= 5.0:
        labels.append("Strong 1m confirm")

    if event_vol >= 3.0 and event_range >= 1.5 and event_relative >= 1.2:
        labels.append("Strong 5m event")

    if (
        confirm_vol >= 2.5
        and confirm_body_bps >= 5.0
        and event_vol >= 3.0
        and event_range >= 1.5
        and event_relative >= 1.2
    ):
        labels.append("Dual strong")

    return labels


def run_entry_quality_lab(
    rolling_details: pd.DataFrame,
    *,
    forward_days: int,
    fee_bps: float,
    slippage_bps: float,
    max_periods: int = 12,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Diagnose whether stronger pre-entry evidence improves the frozen 10m signal.

    Signal family stays fixed:
      - LONG
      - alt-basket BULL regime
      - 5m relative breakout event
      - 1m same-direction volume confirmation
      - next 1m open entry
      - fixed 10-minute exit

    Only coarse, pre-declared quality buckets are compared.
    """
    if rolling_details is None or rolling_details.empty:
        raise ValueError("Run Rolling Universe Validation first.")

    selected = rolling_details[["Period", "Selection time", "Symbol"]].drop_duplicates().copy()
    selected["Selection time"] = pd.to_datetime(selected["Selection time"], utc=True)
    keep_periods = sorted(selected["Period"].unique())[-int(max_periods):]
    selected = selected[selected["Period"].isin(keep_periods)]

    cost_bps = 2.0 * (float(fee_bps) + float(slippage_bps))
    rows = []

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
                if entry_i is None or entry_i <= 0:
                    continue

                exit_i = entry_i + MAX_HOLD_MINUTES - 1
                if exit_i >= len(one_minute):
                    continue

                entry_price = float(one_minute.iloc[entry_i]["open"])
                exit_price = float(one_minute.iloc[exit_i]["close"])
                if entry_price <= 0:
                    continue

                confirm_row = one_minute.iloc[entry_i - 1]
                gross_bps = (exit_price / entry_price - 1.0) * 10_000.0
                confirm_body_bps = (
                    (float(confirm_row["close"]) / float(confirm_row["open"]) - 1.0)
                    * 10_000.0
                    if float(confirm_row["open"]) > 0
                    else 0.0
                )

                for tier in _tiers(event, confirm_row):
                    rows.append(
                        {
                            "Tier": tier,
                            "Period": int(period),
                            "Selection time": selection_time,
                            "Symbol": symbol,
                            "Event time": event_time,
                            "Entry time": one_minute.index[entry_i],
                            "Gross bps": gross_bps,
                            "Net bps": gross_bps - cost_bps,
                            "1m confirm volume ratio": float(confirm_row["volume_ratio_1m"]),
                            "1m confirm body bps": confirm_body_bps,
                            "5m volume ratio": float(event["coin_volume_ratio"]),
                            "5m range / ATR": float(event["range_atr"]),
                            "5m relative move / ATR": float(event["relative_move_atr"]),
                            "Breadth": float(event["breadth"]),
                        }
                    )

    details = pd.DataFrame(rows)
    if details.empty:
        raise RuntimeError("No entry-quality events were available.")

    summary_rows = []
    for tier, group in details.groupby("Tier", sort=False):
        gross = group["Gross bps"].astype(float)
        net = group["Net bps"].astype(float)
        period_avg = group.groupby("Period")["Net bps"].mean()

        summary_rows.append(
            {
                "Tier": tier,
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
                "Avg 1m confirm vol": float(group["1m confirm volume ratio"].mean()),
                "Avg 1m body bps": float(group["1m confirm body bps"].mean()),
                "Avg 5m event vol": float(group["5m volume ratio"].mean()),
                "Avg 5m relative / ATR": float(group["5m relative move / ATR"].mean()),
                "Round-trip cost bps": cost_bps,
            }
        )

    return pd.DataFrame(summary_rows), details
