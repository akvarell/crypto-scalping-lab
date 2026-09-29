from __future__ import annotations

import pandas as pd

from src.rolling_universe import (
    RESEARCH_ANCHOR_UTC,
    _all_current_usdt_symbols,
    _fetch_hourly_history,
    _historical_liquidity_pools,
    _rank_score,
    _window_metrics,
)


def build_research_universe_v3(
    *,
    horizon_days: int = 90,
    lookback_days: int = 14,
    forward_days: int = 7,
    pool_size: int = 20,
    trade_top_n: int = 5,
    context_top_n: int = 15,
    min_daily_turnover_usd: float = 5_000_000.0,
    end_offset_days: int = 0,
    as_of: str | pd.Timestamp | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    """Build a reproducible trade/context universe for v3 research.

    Trade universe and market-context universe are separated:
    - top N trade symbols are the only symbols eligible for entries
    - a wider ranked context basket is used for breadth/regime features

    Important limitation: Binance's standard public exchangeInfo endpoint is
    current-state only. Therefore this historical universe is conditional on
    symbols that are still listed today. v3 reports this explicitly instead
    of treating the universe as survivorship-free.
    """
    horizon_days = max(30, min(int(horizon_days), 365))
    lookback_days = max(7, min(int(lookback_days), 60))
    forward_days = max(1, min(int(forward_days), 14))
    pool_size = max(10, min(int(pool_size), 50))
    trade_top_n = max(1, min(int(trade_top_n), pool_size))
    context_top_n = max(trade_top_n, min(int(context_top_n), pool_size))
    end_offset_days = max(0, min(int(end_offset_days), 730))

    if as_of is None:
        anchor = RESEARCH_ANCHOR_UTC
    else:
        anchor = pd.Timestamp(as_of)
        if anchor.tzinfo is None:
            anchor = anchor.tz_localize("UTC")
        else:
            anchor = anchor.tz_convert("UTC")
        anchor = anchor.floor("h")

    now = anchor - pd.Timedelta(days=end_offset_days)
    analysis_start = now - pd.Timedelta(days=horizon_days)
    history_start = analysis_start - pd.Timedelta(days=lookback_days + 2)

    selection_times: list[pd.Timestamp] = []
    t = analysis_start
    while t + pd.Timedelta(days=forward_days) <= now:
        selection_times.append(t)
        t += pd.Timedelta(days=forward_days)

    current_symbols = _all_current_usdt_symbols()
    start_ms = int(history_start.timestamp() * 1000)
    end_ms = int(now.timestamp() * 1000)

    pools = _historical_liquidity_pools(
        current_symbols,
        selection_times,
        lookback_days,
        pool_size,
        min_daily_turnover_usd,
        start_ms,
        end_ms,
    )

    union_symbols = sorted({s for values in pools.values() for s in values})
    history: dict[str, pd.DataFrame] = {}
    for symbol in union_symbols:
        try:
            history[symbol] = _fetch_hourly_history(symbol, start_ms, end_ms)
        except Exception:
            history[symbol] = pd.DataFrame()

    rank_rows: list[dict] = []
    period_rows: list[dict] = []

    for period_number, selection_time in enumerate(selection_times, start=1):
        lookback_start = selection_time - pd.Timedelta(days=lookback_days)
        forward_end = selection_time + pd.Timedelta(days=forward_days)
        candidates = pools.get(selection_time, [])

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

        if len(sample_rows) < max(3, trade_top_n):
            continue

        ranked = (
            _rank_score(pd.DataFrame(sample_rows))
            .sort_values("score", ascending=False)
            .reset_index(drop=True)
        )
        ranked["Rank"] = ranked.index + 1
        ranked["Trade selected"] = ranked["Rank"] <= trade_top_n
        ranked["Context selected"] = ranked["Rank"] <= context_top_n

        # Forward metrics are diagnostic only and never used for ranking.
        for _, row in ranked.iterrows():
            symbol = str(row["Symbol"])
            df = history.get(symbol, pd.DataFrame())
            forward = df[(df.index >= selection_time) & (df.index < forward_end)]
            forward_metrics = _window_metrics(forward)

            rank_rows.append(
                {
                    "Period": int(period_number),
                    "Selection time": selection_time,
                    "Symbol": symbol,
                    "Rank": int(row["Rank"]),
                    "Trade selected": bool(row["Trade selected"]),
                    "Context selected": bool(row["Context selected"]),
                    "Selection score": float(row["score"]),
                    "Lookback vol %": float(row["realized_vol_pct"]),
                    "Lookback ATR %": float(row["atr_pct"]),
                    "Lookback daily turnover $": float(row["avg_daily_turnover"]),
                    "Lookback trades/day": float(row["avg_daily_trades"]),
                    "Forward vol %": (
                        float(forward_metrics["realized_vol_pct"])
                        if forward_metrics is not None
                        else float("nan")
                    ),
                    "Forward abs 1h move %": (
                        float(forward_metrics["avg_abs_1h_pct"])
                        if forward_metrics is not None
                        else float("nan")
                    ),
                }
            )

        selected = ranked[ranked["Trade selected"]]
        context = ranked[ranked["Context selected"]]
        period_rows.append(
            {
                "Period": int(period_number),
                "Selection time": selection_time,
                "Eligible symbols": int(len(ranked)),
                "Trade symbols": int(len(selected)),
                "Context symbols": int(len(context)),
                "Trade list": ", ".join(selected["Symbol"].astype(str)),
                "Context list": ", ".join(context["Symbol"].astype(str)),
            }
        )

    periods = pd.DataFrame(period_rows)
    ranked_details = pd.DataFrame(rank_rows)

    summary = {
        "periods": int(len(periods)),
        "trade_top_n": int(trade_top_n),
        "context_top_n": int(context_top_n),
        "current_symbol_master_count": int(len(current_symbols)),
        "survivorship_status": "CURRENT_LISTED_ONLY",
        "survivorship_note": (
            "Historical ranking is conditional on symbols that are still listed on Binance today. "
            "Delisted historical symbols are not reconstructed by the standard public API."
        ),
    }

    return periods, ranked_details, summary
