from __future__ import annotations

import math

import pandas as pd

from src.event_dataset_v3 import build_event_dataset_v3
from src.rolling_universe import (
    _all_current_usdt_symbols,
    _fetch_hourly_history,
    _historical_liquidity_pools,
    _rank_score,
    _window_metrics,
)
from src.validation_v3 import cost_stress_v3


# Clean forward start chosen after the historical research program was stopped.
FORWARD_START_UTC = pd.Timestamp("2026-10-01T00:00:00Z")
MIN_MATURE_MINUTES = 20


def _normalize_utc(value: str | pd.Timestamp) -> pd.Timestamp:
    ts = pd.Timestamp(value)
    if ts.tzinfo is None:
        return ts.tz_localize("UTC")
    return ts.tz_convert("UTC")


def build_forward_universe_v4(
    *,
    selection_time: str | pd.Timestamp = FORWARD_START_UTC,
    lookback_days: int = 14,
    pool_size: int = 20,
    trade_top_n: int = 5,
    context_top_n: int = 15,
    min_daily_turnover_usd: float = 5_000_000.0,
) -> tuple[pd.DataFrame, dict]:
    """Freeze one forward trade/context universe using only pre-start data."""
    selection_time = _normalize_utc(selection_time)
    lookback_days = max(7, min(int(lookback_days), 60))
    pool_size = max(10, min(int(pool_size), 50))
    trade_top_n = max(1, min(int(trade_top_n), pool_size))
    context_top_n = max(trade_top_n, min(int(context_top_n), pool_size))

    history_start = selection_time - pd.Timedelta(days=lookback_days + 2)
    current_symbols = _all_current_usdt_symbols()

    start_ms = int(history_start.timestamp() * 1000)
    end_ms = int(selection_time.timestamp() * 1000)

    pools = _historical_liquidity_pools(
        current_symbols,
        [selection_time],
        lookback_days,
        pool_size,
        min_daily_turnover_usd,
        start_ms,
        end_ms,
    )
    candidates = pools.get(selection_time, [])
    if len(candidates) < trade_top_n:
        raise RuntimeError("Not enough liquid symbols for the v4 forward universe.")

    history: dict[str, pd.DataFrame] = {}
    for symbol in candidates:
        try:
            history[symbol] = _fetch_hourly_history(symbol, start_ms, end_ms)
        except Exception:
            history[symbol] = pd.DataFrame()

    lookback_start = selection_time - pd.Timedelta(days=lookback_days)
    sample_rows = []
    for symbol in candidates:
        df = history.get(symbol, pd.DataFrame())
        if df.empty:
            continue
        lookback = df[(df.index >= lookback_start) & (df.index < selection_time)]
        metrics = _window_metrics(lookback)
        if metrics is None:
            continue
        sample_rows.append({"Symbol": symbol, **metrics})

    if len(sample_rows) < context_top_n:
        raise RuntimeError(
            f"Only {len(sample_rows)} symbols have enough pre-start history; "
            f"{context_top_n} are required for the context basket."
        )

    ranked = (
        _rank_score(pd.DataFrame(sample_rows))
        .sort_values("score", ascending=False)
        .reset_index(drop=True)
    )
    ranked["Rank"] = ranked.index + 1
    ranked["Trade selected"] = ranked["Rank"] <= trade_top_n
    ranked["Context selected"] = ranked["Rank"] <= context_top_n
    ranked["Period"] = 1
    ranked["Selection time"] = selection_time

    details = ranked[
        [
            "Period",
            "Selection time",
            "Symbol",
            "Rank",
            "Trade selected",
            "Context selected",
            "score",
            "realized_vol_pct",
            "atr_pct",
            "avg_daily_turnover",
            "avg_daily_trades",
        ]
    ].rename(
        columns={
            "score": "Selection score",
            "realized_vol_pct": "Lookback vol %",
            "atr_pct": "Lookback ATR %",
            "avg_daily_turnover": "Lookback daily turnover $",
            "avg_daily_trades": "Lookback trades/day",
        }
    )

    summary = {
        "selection_time": selection_time,
        "trade_list": ranked.loc[
            ranked["Trade selected"], "Symbol"
        ].astype(str).tolist(),
        "context_list": ranked.loc[
            ranked["Context selected"], "Symbol"
        ].astype(str).tolist(),
        "eligible_symbols": int(len(ranked)),
        "symbol_master_status": "CURRENT_LISTED_AT_OBSERVER_RUN",
    }
    return details, summary


def _forward_event_summary(events: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for side in ["ALL", "LONG", "SHORT"]:
        group = events if side == "ALL" else events[events["Side"] == side]
        if group.empty:
            continue

        row = {
            "Side": side,
            "Events": int(len(group)),
            "Symbols": int(group["Symbol"].nunique()),
        }
        for horizon in [3, 5, 10, 15]:
            col = f"Gross {horizon}m bps"
            values = pd.to_numeric(group[col], errors="coerce").dropna()
            row[f"{horizon}m gross avg bps"] = (
                float(values.mean()) if not values.empty else float("nan")
            )
            row[f"{horizon}m gross median bps"] = (
                float(values.median()) if not values.empty else float("nan")
            )
        rows.append(row)

    return pd.DataFrame(rows)


def run_forward_observer_v4(
    *,
    now: str | pd.Timestamp | None = None,
    lookback_days: int = 14,
    pool_size: int = 20,
    trade_top_n: int = 5,
    context_top_n: int = 15,
    min_daily_turnover_usd: float = 5_000_000.0,
) -> dict:
    """Reconstruct paper-only forward observations after a frozen future start."""
    now = pd.Timestamp.now(tz="UTC") if now is None else _normalize_utc(now)
    mature_cutoff = now.floor("min") - pd.Timedelta(minutes=MIN_MATURE_MINUTES)

    if mature_cutoff <= FORWARD_START_UTC:
        wait_minutes = max(
            0,
            int((FORWARD_START_UTC - mature_cutoff).total_seconds() // 60),
        )
        return {
            "status": "WAITING",
            "message": (
                f"Forward observer starts at {FORWARD_START_UTC.isoformat()}. "
                f"Need about {wait_minutes} more minutes before the first fully matured outcomes."
            ),
            "start": FORWARD_START_UTC,
            "cutoff": mature_cutoff,
        }

    universe, universe_summary = build_forward_universe_v4(
        selection_time=FORWARD_START_UTC,
        lookback_days=lookback_days,
        pool_size=pool_size,
        trade_top_n=trade_top_n,
        context_top_n=context_top_n,
        min_daily_turnover_usd=min_daily_turnover_usd,
    )

    elapsed_days = max(
        1,
        int(math.ceil((mature_cutoff - FORWARD_START_UTC).total_seconds() / 86_400.0)),
    )

    events = build_event_dataset_v3(
        universe,
        forward_days=elapsed_days + 1,
        warmup_hours=48,
        period_end_override=mature_cutoff,
    )

    if events.empty:
        return {
            "status": "OBSERVING",
            "message": (
                "The forward window is active, but no broad causal breakout event has matured yet. "
                "No thresholds are changed and no strategy conclusion is allowed."
            ),
            "start": FORWARD_START_UTC,
            "cutoff": mature_cutoff,
            "universe_summary": universe_summary,
            "events": events,
            "event_summary": pd.DataFrame(),
            "cost_stress": pd.DataFrame(),
        }

    event_summary = _forward_event_summary(events)
    cost_table = cost_stress_v3(events)

    event_count = int(len(events))
    calendar_days = float(
        (mature_cutoff - FORWARD_START_UTC).total_seconds() / 86_400.0
    )

    if event_count >= 100 and calendar_days >= 7:
        status = "EVALUATABLE"
        message = (
            f"Forward observer has {event_count} matured events over {calendar_days:.1f} days. "
            "The sample is large enough for a first descriptive review, but no historical threshold "
            "may be retuned from these results."
        )
    else:
        status = "OBSERVING"
        message = (
            f"Forward observer has {event_count} matured events over {calendar_days:.1f} days. "
            "Keep collecting. A first serious review waits for at least 100 events and 7 calendar days."
        )

    return {
        "status": status,
        "message": message,
        "start": FORWARD_START_UTC,
        "cutoff": mature_cutoff,
        "universe_summary": universe_summary,
        "events": events,
        "event_summary": event_summary,
        "cost_stress": cost_table,
    }
