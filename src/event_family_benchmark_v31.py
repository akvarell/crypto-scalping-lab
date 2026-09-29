from __future__ import annotations

import numpy as np
import pandas as pd

from src.event_dataset_v3 import (
    _context_for_trade_symbol,
    _forward_return_bps,
    _merge_micro_windows,
    _one_minute_features,
    _prepare_5m_v3,
)
from src.dynamic_scalping import _fetch_5m
from src.scalping_edge_map import _fetch_1m


FAMILIES = [
    "Breakout continuation",
    "Failed breakout reversal",
    "Pullback reclaim",
    "Cross-sectional leader/laggard",
]

HORIZONS = [3, 5, 10, 15]


def _cooldown(frame: pd.DataFrame, minutes: int = 30) -> pd.DataFrame:
    if frame.empty:
        return frame

    ordered = frame.sort_index().copy()
    keep_positions = []
    last_by_family_direction: dict[tuple[str, int], pd.Timestamp] = {}

    for pos, (ts, row) in enumerate(ordered.iterrows()):
        key = (str(row["Family"]), int(row["Direction"]))
        last = last_by_family_direction.get(key)
        if last is None or ts - last >= pd.Timedelta(minutes=minutes):
            keep_positions.append(pos)
            last_by_family_direction[key] = ts

    return ordered.iloc[keep_positions].copy()


def _cross_section_rank(
    symbol: str,
    prepared: dict[str, pd.DataFrame],
) -> pd.Series:
    ret15 = pd.concat(
        {s: prepared[s]["ret_15m"] for s in prepared},
        axis=1,
    )
    ranks = ret15.rank(axis=1, pct=True, method="average")
    if symbol not in ranks:
        return pd.Series(dtype=float)
    return ranks[symbol].reindex(prepared[symbol].index)


def _pullback_reclaim_events(
    aligned: pd.DataFrame,
    *,
    max_age_bars: int = 6,
) -> tuple[pd.Series, pd.Series]:
    breakout_long = (
        (aligned["close"] > aligned["prior_high_20"])
        & (aligned["close"].shift(1) <= aligned["prior_high_20"].shift(1))
        & (aligned["rel_ret_15m"] > 0)
    )
    breakout_short = (
        (aligned["close"] < aligned["prior_low_20"])
        & (aligned["close"].shift(1) >= aligned["prior_low_20"].shift(1))
        & (aligned["rel_ret_15m"] < 0)
    )

    long_reclaim = pd.Series(False, index=aligned.index)
    short_reclaim = pd.Series(False, index=aligned.index)

    for i in range(len(aligned)):
        start = max(0, i - int(max_age_bars))
        if start >= i:
            continue

        prior = aligned.iloc[start:i]

        long_hits = np.flatnonzero(
            breakout_long.iloc[start:i].fillna(False).to_numpy()
        )
        if len(long_hits):
            j = start + int(long_hits[-1])
            level = float(aligned.iloc[j]["prior_high_20"])
            row = aligned.iloc[i]
            if (
                pd.notna(level)
                and float(row["low"]) <= level
                and float(row["close"]) > level
                and float(row["close"]) > float(row["open"])
            ):
                long_reclaim.iloc[i] = True

        short_hits = np.flatnonzero(
            breakout_short.iloc[start:i].fillna(False).to_numpy()
        )
        if len(short_hits):
            j = start + int(short_hits[-1])
            level = float(aligned.iloc[j]["prior_low_20"])
            row = aligned.iloc[i]
            if (
                pd.notna(level)
                and float(row["high"]) >= level
                and float(row["close"]) < level
                and float(row["close"]) < float(row["open"])
            ):
                short_reclaim.iloc[i] = True

    return long_reclaim, short_reclaim


def _family_events(
    symbol: str,
    coin: pd.DataFrame,
    ctx: pd.DataFrame,
    prepared: dict[str, pd.DataFrame],
    *,
    selection_time: pd.Timestamp,
    period_end: pd.Timestamp,
) -> pd.DataFrame:
    aligned = coin.join(ctx, how="left")
    aligned["rel_ret_15m"] = aligned["ret_15m"] - aligned["basket_ret_15m"]
    aligned["relative_move_atr"] = (
        aligned["rel_ret_15m"].abs()
        / aligned["atr_pct"].replace(0.0, float("nan"))
    )
    aligned["xs_rank_15m"] = _cross_section_rank(symbol, prepared)

    rows = []

    breakout_long = (
        (aligned["close"] > aligned["prior_high_20"])
        & (aligned["close"].shift(1) <= aligned["prior_high_20"].shift(1))
        & (aligned["rel_ret_15m"] > 0)
    )
    breakout_short = (
        (aligned["close"] < aligned["prior_low_20"])
        & (aligned["close"].shift(1) >= aligned["prior_low_20"].shift(1))
        & (aligned["rel_ret_15m"] < 0)
    )

    failed_high = (
        (aligned["high"] > aligned["prior_high_20"])
        & (aligned["close"] < aligned["prior_high_20"])
        & (aligned["close"] < aligned["open"])
        & (aligned["rel_ret_15m"] > 0)
    )
    failed_low = (
        (aligned["low"] < aligned["prior_low_20"])
        & (aligned["close"] > aligned["prior_low_20"])
        & (aligned["close"] > aligned["open"])
        & (aligned["rel_ret_15m"] < 0)
    )

    reclaim_long, reclaim_short = _pullback_reclaim_events(aligned)

    laggard_long = (
        (aligned["xs_rank_15m"] <= 0.20)
        & (aligned["basket_ret_60m"] > 0)
        & (aligned["close"] > aligned["open"])
    )
    leader_short = (
        (aligned["xs_rank_15m"] >= 0.80)
        & (aligned["basket_ret_60m"] < 0)
        & (aligned["close"] < aligned["open"])
    )

    specs = [
        ("Breakout continuation", breakout_long, 1),
        ("Breakout continuation", breakout_short, -1),
        ("Failed breakout reversal", failed_low, 1),
        ("Failed breakout reversal", failed_high, -1),
        ("Pullback reclaim", reclaim_long, 1),
        ("Pullback reclaim", reclaim_short, -1),
        ("Cross-sectional leader/laggard", laggard_long, 1),
        ("Cross-sectional leader/laggard", leader_short, -1),
    ]

    for family, mask, direction in specs:
        idx = aligned.index[
            mask.fillna(False)
            & (aligned.index >= selection_time)
            & (aligned.index < period_end)
        ]
        for ts in idx:
            row = aligned.loc[ts]
            rows.append(
                {
                    "Event time": ts,
                    "Family": family,
                    "Direction": int(direction),
                    "Breadth": float(row["breadth"]),
                    "Basket 15m return %": float(row["basket_ret_15m"]) * 100.0,
                    "Basket 60m return %": float(row["basket_ret_60m"]) * 100.0,
                    "Basket volume ratio": float(row["basket_volume_ratio"]),
                    "5m volume ratio prior20": float(row["volume_ratio"]),
                    "5m trades ratio prior20": float(
                        row.get("trades_ratio", float("nan"))
                    ),
                    "5m range / ATR": float(row["range_atr"]),
                    "5m relative move / ATR": float(row["relative_move_atr"]),
                    "5m ATR %": float(row["atr_pct"]) * 100.0,
                    "5m RSI": float(row["rsi"]),
                    "Cross-sectional 15m rank": float(row["xs_rank_15m"]),
                    "5m taker buy ratio": float(
                        row.get("taker_buy_ratio", float("nan"))
                    ),
                    "5m taker buy ratio delta": float(
                        row.get("taker_buy_ratio_delta", float("nan"))
                    ),
                }
            )

    if not rows:
        return pd.DataFrame()

    out = pd.DataFrame(rows).set_index("Event time").sort_index()
    return _cooldown(out, minutes=30)


def build_event_family_period_v31(
    period_frame: pd.DataFrame,
    *,
    forward_days: int = 7,
    warmup_hours: int = 48,
) -> pd.DataFrame:
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

    output_rows = []

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

        for batch in _merge_micro_windows(list(event_lookup)):
            try:
                one_minute_batch = _fetch_1m(
                    symbol,
                    batch["start"],
                    batch["end"],
                )
            except Exception:
                one_minute_batch = pd.DataFrame()

            if one_minute_batch.empty:
                continue

            for event_time in batch["events"]:
                known_time = pd.Timestamp(event_time) + pd.Timedelta(minutes=5)
                micro_start = known_time - pd.Timedelta(minutes=30)
                micro_end = known_time + pd.Timedelta(minutes=20)

                one_minute = one_minute_batch[
                    (one_minute_batch.index >= micro_start)
                    & (one_minute_batch.index < micro_end)
                ]
                if one_minute.empty or len(one_minute) < 35:
                    continue

                decision_i = int(
                    one_minute.index.searchsorted(known_time, side="left")
                )
                if decision_i >= len(one_minute):
                    continue

                entry_i = decision_i + 1
                if entry_i >= len(one_minute):
                    continue

                one_features = _one_minute_features(one_minute, decision_i)

                five_row = coin.loc[event_time]
                candle_range = float(five_row["high"]) - float(five_row["low"])
                close_location = (
                    (float(five_row["close"]) - float(five_row["low"]))
                    / candle_range
                    if candle_range > 0
                    else float("nan")
                )

                for event in event_lookup[event_time]:
                    record = {
                        "Period": period,
                        "Selection time": selection_time,
                        "Symbol": symbol,
                        "Family": str(event["Family"]),
                        "Event time": event_time,
                        "Known time": known_time,
                        "Entry time": one_minute.index[entry_i],
                        "Side": (
                            "LONG"
                            if int(event["Direction"]) == 1
                            else "SHORT"
                        ),
                        "Direction": int(event["Direction"]),
                        "5m close location": close_location,
                    }

                    for col in [
                        "Breadth",
                        "Basket 15m return %",
                        "Basket 60m return %",
                        "Basket volume ratio",
                        "5m volume ratio prior20",
                        "5m trades ratio prior20",
                        "5m range / ATR",
                        "5m relative move / ATR",
                        "5m ATR %",
                        "5m RSI",
                        "Cross-sectional 15m rank",
                        "5m taker buy ratio",
                        "5m taker buy ratio delta",
                    ]:
                        record[col] = event.get(col, float("nan"))

                    record.update(one_features)

                    for horizon in HORIZONS:
                        record[f"Gross {horizon}m bps"] = _forward_return_bps(
                            one_minute,
                            entry_i,
                            int(event["Direction"]),
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
    cost_bps: float,
    repeats: int = 500,
    seed: int = 43,
) -> tuple[float, float]:
    periods = sorted(frame["Period"].dropna().unique().tolist())
    if len(periods) < 6:
        return float("nan"), float("nan")

    rng = np.random.default_rng(seed)
    values = []

    for _ in range(int(repeats)):
        sampled = rng.choice(periods, size=len(periods), replace=True)
        parts = []
        for period in sampled:
            part = pd.to_numeric(
                frame.loc[frame["Period"] == period, "Gross 10m bps"],
                errors="coerce",
            ).dropna()
            if not part.empty:
                parts.append(part)

        if not parts:
            continue

        gross = pd.concat(parts, ignore_index=True)
        values.append(float((gross - float(cost_bps)).mean()))

    if len(values) < 100:
        return float("nan"), float("nan")

    return (
        float(np.quantile(values, 0.005)),
        float(np.quantile(values, 0.995)),
    )


def benchmark_event_families_v31(
    events: pd.DataFrame,
    *,
    costs_bps: tuple[float, ...] = (12.0, 20.0, 30.0),
) -> tuple[pd.DataFrame, str]:
    if events is None or events.empty:
        raise ValueError("v3.1 event family dataset is empty.")

    rows = []

    for family in FAMILIES:
        family_frame = events[events["Family"] == family]
        if family_frame.empty:
            continue

        for side in ["ALL", "LONG", "SHORT"]:
            frame = (
                family_frame
                if side == "ALL"
                else family_frame[family_frame["Side"] == side]
            )
            if frame.empty:
                continue

            gross = pd.to_numeric(frame["Gross 10m bps"], errors="coerce")
            valid = frame.loc[gross.notna()].copy()
            valid["Gross 10m bps"] = gross.dropna().values
            if valid.empty:
                continue

            top_symbol_share = (
                float(valid["Symbol"].value_counts().iloc[0] / len(valid) * 100.0)
                if len(valid)
                else 0.0
            )

            for cost in costs_bps:
                net = valid["Gross 10m bps"] - float(cost)
                period_avg = valid.assign(_net=net).groupby("Period")["_net"].mean()
                ci_low, ci_high = _period_bootstrap_net_ci(
                    valid,
                    cost_bps=float(cost),
                )

                candidate = (
                    float(cost) == 12.0
                    and len(valid) >= 60
                    and valid["Period"].nunique() >= 8
                    and valid["Symbol"].nunique() >= 8
                    and float(net.mean()) > 0
                    and float(net.median()) > 0
                    and _trimmed_mean(valid["Gross 10m bps"]) > float(cost)
                    and int((period_avg > 0).sum()) > len(period_avg) / 2
                    and top_symbol_share <= 35.0
                    and np.isfinite(ci_low)
                    and ci_low > 0
                )

                rows.append(
                    {
                        "Family": family,
                        "Side": side,
                        "Round-trip cost bps": float(cost),
                        "Events": int(len(valid)),
                        "Periods": int(valid["Period"].nunique()),
                        "Symbols": int(valid["Symbol"].nunique()),
                        "Top symbol share %": top_symbol_share,
                        "Gross avg bps": float(valid["Gross 10m bps"].mean()),
                        "Gross median bps": float(valid["Gross 10m bps"].median()),
                        "Trimmed gross avg bps": _trimmed_mean(valid["Gross 10m bps"]),
                        "Net avg bps": float(net.mean()),
                        "Net median bps": float(net.median()),
                        "Positive periods": (
                            f"{int((period_avg > 0).sum())}/{len(period_avg)}"
                            if len(period_avg)
                            else "0/0"
                        ),
                        "99% cluster net CI low": ci_low,
                        "99% cluster net CI high": ci_high,
                        "Family candidate": "YES" if candidate else "NO",
                    }
                )

    summary = pd.DataFrame(rows)
    if summary.empty:
        raise RuntimeError("v3.1 benchmark produced no comparable family results.")

    base12 = summary[summary["Round-trip cost bps"] == 12.0].copy()
    candidates = base12[base12["Family candidate"] == "YES"]

    if candidates.empty:
        best = (
            base12.sort_values("Net avg bps", ascending=False).iloc[0]
            if not base12.empty
            else None
        )
        if best is None:
            verdict = "No v3.1 family could be evaluated."
        else:
            verdict = (
                "No event family passed the strict 12 bps candidate screen. "
                f"The strongest descriptive group was {best['Family']} / {best['Side']} "
                f"with net avg {float(best['Net avg bps']):+.1f} bps, net median "
                f"{float(best['Net median bps']):+.1f} bps and positive periods "
                f"{best['Positive periods']}. Treat this only as diagnostic, not as a strategy."
            )
    else:
        descriptions = []
        for _, row in candidates.head(3).iterrows():
            descriptions.append(
                f"{row['Family']} / {row['Side']} "
                f"(events {int(row['Events'])}, net avg {float(row['Net avg bps']):+.1f} bps, "
                f"median {float(row['Net median bps']):+.1f}, periods {row['Positive periods']}, "
                f"99% CI low {float(row['99% cluster net CI low']):+.1f})"
            )
        verdict = (
            "v3.1 found event-family candidates that remained positive after 12 bps costs "
            "and strict period-cluster checks: "
            + "; ".join(descriptions)
            + ". They still require purged walk-forward validation before any forward paper test."
        )

    return summary, verdict
