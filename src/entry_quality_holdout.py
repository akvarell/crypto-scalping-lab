from __future__ import annotations

import pandas as pd

from src.alt_basket_study import _alt_context, _cooldown, _events, _prepare_coin
from src.dynamic_scalping import _fetch_5m
from src.rolling_universe import run_rolling_universe_validation
from src.scalping_edge_map import _add_1m_micro_features, _entry_index_confirmed, _fetch_1m


FROZEN_TIERS = [
    "Baseline LONG/BULL",
    "Strong 1m confirm",
    "Dual strong",
]

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


def _tier_labels(event: pd.Series, confirm_row: pd.Series) -> list[str]:
    """Frozen v2.2 thresholds. Do not tune inside this holdout."""
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

    labels = ["Baseline LONG/BULL"]

    strong_1m = confirm_vol >= 2.5 and confirm_body_bps >= 5.0
    strong_5m = (
        event_vol >= 3.0
        and event_range >= 1.5
        and event_relative >= 1.2
    )

    if strong_1m:
        labels.append("Strong 1m confirm")
    if strong_1m and strong_5m:
        labels.append("Dual strong")

    return labels


def _evaluate_frozen_entries(
    rolling_details: pd.DataFrame,
    *,
    forward_days: int,
    fee_bps: float,
    slippage_bps: float,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    cost_bps = 2.0 * (float(fee_bps) + float(slippage_bps))
    rows = []

    for period, period_sel in rolling_details.groupby("Period", sort=True):
        selection_time = pd.Timestamp(period_sel["Selection time"].iloc[0])
        period_end = selection_time + pd.Timedelta(days=int(forward_days))

        prepared: dict[str, pd.DataFrame] = {}

        # Heavy 1m history is NOT downloaded here. First find candidate 5m events.
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
                # 5m event is known only after the signal candle closes.
                known_time = pd.Timestamp(event_time) + pd.Timedelta(minutes=5)

                # Pull 30m of prehistory so the 1m 20-bar volume average is
                # already available at confirmation time, plus enough future
                # data for the 3m confirmation window and fixed 10m exit.
                micro_start = known_time - pd.Timedelta(minutes=30)
                micro_end = known_time + pd.Timedelta(minutes=20)

                try:
                    one_minute = _fetch_1m(symbol, micro_start, micro_end)
                except Exception:
                    one_minute = pd.DataFrame()

                if one_minute.empty or len(one_minute) < 35:
                    continue

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
                gross_bps = (exit_price / entry_price - 1.0) * 10_000.0
                confirm_body_bps = (
                    (float(confirm_row["close"]) / float(confirm_row["open"]) - 1.0)
                    * 10_000.0
                    if float(confirm_row["open"]) > 0
                    else 0.0
                )

                for tier in _tier_labels(event, confirm_row):
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
        raise RuntimeError("No frozen 1m entry events appeared in the older holdout.")

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

    summary = pd.DataFrame(summary_rows)
    order = {name: i for i, name in enumerate(FROZEN_TIERS)}
    summary["_order"] = summary["Tier"].map(order)
    summary = summary.sort_values("_order").drop(columns=["_order"]).reset_index(drop=True)

    return summary, details


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

    The window ends 270 days before today so it stays separate from the recent
    development sample and the previously examined v1.9 holdout.

    Optimized implementation: historical universe and 5m event detection are
    done first; 1m data is downloaded only around candidate events.
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

    summary, details = _evaluate_frozen_entries(
        rolling_details,
        forward_days=int(forward_days),
        fee_bps=float(fee_bps),
        slippage_bps=float(slippage_bps),
    )

    return summary, details, rolling_periods, rolling_summary
