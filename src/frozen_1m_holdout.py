from __future__ import annotations

import pandas as pd

from src.alt_basket_study import _alt_context, _cooldown, _events, _prepare_coin
from src.dynamic_scalping import _fetch_5m
from src.rolling_universe import run_rolling_universe_validation
from src.scalping_edge_map import _add_1m_micro_features, _entry_index_confirmed, _fetch_1m


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


def _classify_frozen_tiers(event: pd.Series, confirm_row: pd.Series) -> list[str]:
    """Frozen v2.2 thresholds. Do not tune inside the holdout."""
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

    strong_1m = confirm_vol >= 2.5 and confirm_body_bps >= 5.0
    strong_5m = (
        event_vol >= 3.0
        and event_range >= 1.5
        and event_relative >= 1.2
    )

    labels = []
    if strong_1m:
        labels.append("Strong 1m confirm")
    if strong_1m and strong_5m:
        labels.append("Dual strong")
    return labels


def _evaluate_holdout_events(
    rolling_details: pd.DataFrame,
    *,
    forward_days: int,
    fee_bps: float,
    slippage_bps: float,
) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    cost_bps = 2.0 * (float(fee_bps) + float(slippage_bps))
    rows = []

    candidate_5m_events = 0
    one_minute_windows_loaded = 0

    for period, period_sel in rolling_details.groupby("Period", sort=True):
        selection_time = pd.Timestamp(period_sel["Selection time"].iloc[0])
        period_end = selection_time + pd.Timedelta(days=int(forward_days))

        prepared: dict[str, pd.DataFrame] = {}

        for symbol in period_sel["Symbol"].astype(str).tolist():
            try:
                market_5m = _fetch_5m(symbol, selection_time, period_end)
            except Exception:
                market_5m = pd.DataFrame()

            if market_5m.empty or len(market_5m) < 200:
                continue
            prepared[symbol] = _prepare_coin(market_5m)

        if len(prepared) < 3:
            continue

        contexts = _alt_context(prepared)

        for symbol, five_minute in prepared.items():
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
                candidate_5m_events += 1

                # The 5m event is known only after its candle closes.
                known_time = pd.Timestamp(event_time) + pd.Timedelta(minutes=5)

                # At most 3 completed 1m candles for confirmation, then next 1m
                # entry plus fixed 10-minute hold. 20 minutes is enough.
                micro_end = known_time + pd.Timedelta(minutes=20)

                try:
                    one_minute = _fetch_1m(symbol, known_time, micro_end)
                except Exception:
                    one_minute = pd.DataFrame()

                if one_minute.empty or len(one_minute) < 12:
                    continue

                one_minute_windows_loaded += 1
                one_minute = _add_1m_micro_features(one_minute)

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
                tiers = _classify_frozen_tiers(event, confirm_row)
                if not tiers:
                    continue

                gross_bps = (exit_price / entry_price - 1.0) * 10_000.0
                confirm_body_bps = (
                    (float(confirm_row["close"]) / float(confirm_row["open"]) - 1.0)
                    * 10_000.0
                    if float(confirm_row["open"]) > 0
                    else 0.0
                )

                for tier in tiers:
                    rows.append(
                        {
                            "Tier": tier,
                            "Period": int(period),
                            "Selection time": selection_time,
                            "Symbol": symbol,
                            "5m event time": event_time,
                            "1m entry time": one_minute.index[entry_i],
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
        raise RuntimeError("No frozen Strong-1m or Dual-strong events appeared in the holdout.")

    summary_rows = []
    for tier, group in details.groupby("Tier", sort=False):
        gross = group["Gross bps"].astype(float)
        net = group["Net bps"].astype(float)
        period_avg = group.groupby("Period")["Net bps"].mean()

        summary_rows.append(
            {
                "Tier": tier,
                "Events": int(len(group)),
                "Periods with events": int(group["Period"].nunique()),
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
                "Round-trip cost bps": cost_bps,
                "Gross edge / cost": (
                    float(gross.mean()) / cost_bps
                    if cost_bps > 0
                    else float("inf")
                ),
            }
        )

    diagnostics = {
        "candidate_5m_events": int(candidate_5m_events),
        "one_minute_windows_loaded": int(one_minute_windows_loaded),
    }

    return pd.DataFrame(summary_rows), details, diagnostics


def run_frozen_1m_holdout(
    *,
    horizon_days: int = 180,
    end_offset_days: int = 90,
    lookback_days: int = 14,
    forward_days: int = 7,
    pool_size: int = 20,
    select_top_n: int = 5,
    min_daily_turnover_usd: float = 5_000_000.0,
    fee_bps: float = 4.0,
    slippage_bps: float = 2.0,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, dict, dict]:
    """Frozen v2.2 1m rules on an older non-overlapping historical window."""
    periods, rolling_details, universe_summary = run_rolling_universe_validation(
        horizon_days=int(horizon_days),
        lookback_days=int(lookback_days),
        forward_days=int(forward_days),
        pool_size=int(pool_size),
        select_top_n=int(select_top_n),
        min_daily_turnover_usd=float(min_daily_turnover_usd),
        end_offset_days=int(end_offset_days),
    )

    if rolling_details.empty:
        raise RuntimeError("Historical holdout universe produced no selected symbols.")

    summary, details, diagnostics = _evaluate_holdout_events(
        rolling_details,
        forward_days=int(forward_days),
        fee_bps=float(fee_bps),
        slippage_bps=float(slippage_bps),
    )

    return summary, details, periods, universe_summary, diagnostics
