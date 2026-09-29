from __future__ import annotations

import pandas as pd

from src.causal_indicators import add_indicators_v3
from src.dynamic_scalping import _fetch_5m
from src.scalping_edge_map import _fetch_1m


WARMUP_HOURS = 48
FORWARD_HORIZONS = [3, 5, 10, 15]


def _prepare_5m_v3(df: pd.DataFrame) -> pd.DataFrame:
    out = add_indicators_v3(
        df,
        ema_fast=9,
        ema_slow=21,
        trend_ema=100,
        rsi_period=14,
        atr_period=14,
        volume_period=20,
    )
    out["ret_15m"] = out["close"].pct_change(3)
    out["ret_60m"] = out["close"].pct_change(12)
    out["atr_pct"] = out["atr"] / out["close"].replace(0.0, float("nan"))
    out["range_atr"] = (
        (out["high"] - out["low"])
        / out["atr"].replace(0.0, float("nan"))
    )
    out["above_trend"] = out["close"] > out["trend_ema"]
    out["prior_high_20"] = out["high"].rolling(20, min_periods=20).max().shift(1)
    out["prior_low_20"] = out["low"].rolling(20, min_periods=20).min().shift(1)
    return out


def _context_for_trade_symbol(
    symbol: str,
    prepared: dict[str, pd.DataFrame],
) -> pd.DataFrame:
    others = [s for s in prepared if s != symbol]
    if len(others) < 5:
        return pd.DataFrame()

    target_index = prepared[symbol].index

    ret15 = pd.concat(
        {s: prepared[s]["ret_15m"] for s in others},
        axis=1,
    )
    ret60 = pd.concat(
        {s: prepared[s]["ret_60m"] for s in others},
        axis=1,
    )
    breadth = pd.concat(
        {s: prepared[s]["above_trend"].astype(float) for s in others},
        axis=1,
    )
    volume = pd.concat(
        {s: prepared[s]["volume_ratio"] for s in others},
        axis=1,
    )

    ctx = pd.DataFrame(index=target_index)
    ctx["basket_ret_15m"] = ret15.median(axis=1).reindex(target_index)
    ctx["basket_ret_60m"] = ret60.median(axis=1).reindex(target_index)
    ctx["breadth"] = breadth.mean(axis=1).reindex(target_index)
    ctx["basket_volume_ratio"] = volume.median(axis=1).reindex(target_index)
    return ctx


def _candidate_breakouts(
    coin: pd.DataFrame,
    ctx: pd.DataFrame,
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

    long_event = (
        (aligned["close"] > aligned["prior_high_20"])
        & (aligned["close"].shift(1) <= aligned["prior_high_20"].shift(1))
        & (aligned["rel_ret_15m"] > 0)
    )
    short_event = (
        (aligned["close"] < aligned["prior_low_20"])
        & (aligned["close"].shift(1) >= aligned["prior_low_20"].shift(1))
        & (aligned["rel_ret_15m"] < 0)
    )

    out = pd.DataFrame(index=aligned.index)
    out["direction"] = 0
    out.loc[long_event, "direction"] = 1
    out.loc[short_event, "direction"] = -1

    for col in [
        "breadth",
        "basket_ret_15m",
        "basket_ret_60m",
        "basket_volume_ratio",
    ]:
        out[col] = aligned[col]

    out["coin_volume_ratio"] = aligned["volume_ratio"]
    out["coin_trades_ratio"] = aligned.get("trades_ratio")
    out["range_atr"] = aligned["range_atr"]
    out["relative_move_atr"] = aligned["relative_move_atr"]
    out["atr_pct"] = aligned["atr_pct"]
    out["rsi"] = aligned["rsi"]
    out["ema_fast"] = aligned["ema_fast"]
    out["ema_slow"] = aligned["ema_slow"]
    out["trend_ema"] = aligned["trend_ema"]
    out["close"] = aligned["close"]

    if "taker_buy_ratio" in aligned:
        out["5m taker buy ratio"] = aligned["taker_buy_ratio"]
        out["5m taker buy ratio delta"] = aligned["taker_buy_ratio_delta"]

    out = out[
        (out.index >= selection_time)
        & (out.index < period_end)
        & (out["direction"] != 0)
    ].copy()

    return out


def _cooldown_per_symbol(events: pd.DataFrame, minutes: int = 30) -> pd.DataFrame:
    if events.empty:
        return events

    kept = []
    last = None
    for ts in events.index:
        if last is None or ts - last >= pd.Timedelta(minutes=minutes):
            kept.append(ts)
            last = ts
    return events.loc[kept]


def _one_minute_features(
    one_minute: pd.DataFrame,
    decision_i: int,
) -> dict:
    row = one_minute.iloc[decision_i]
    prior20 = one_minute.iloc[max(0, decision_i - 20):decision_i]
    prior5 = one_minute.iloc[max(0, decision_i - 5):decision_i]

    prior_volume = (
        float(prior20["volume"].mean())
        if len(prior20) >= 10
        else float("nan")
    )
    prior_trades = (
        float(prior20["trades"].mean())
        if len(prior20) >= 10 and "trades" in prior20
        else float("nan")
    )

    candle_range = float(row["high"]) - float(row["low"])
    body = float(row["close"]) - float(row["open"])

    taker_ratio = float("nan")
    prior_taker = float("nan")
    if "taker_base" in one_minute.columns and float(row["volume"]) > 0:
        taker_ratio = float(row["taker_base"]) / float(row["volume"])
        if not prior20.empty:
            denom = prior20["volume"].replace(0.0, float("nan"))
            ratios = prior20["taker_base"] / denom
            ratios = ratios.dropna()
            if not ratios.empty:
                prior_taker = float(ratios.mean())

    impulse_3m = float("nan")
    if decision_i >= 3:
        base = float(one_minute.iloc[decision_i - 3]["close"])
        if base > 0:
            impulse_3m = (float(row["close"]) / base - 1.0) * 10_000.0

    realized = prior5["close"].pct_change().dropna()

    return {
        "1m volume ratio prior20": (
            float(row["volume"]) / prior_volume
            if pd.notna(prior_volume) and prior_volume > 0
            else float("nan")
        ),
        "1m trades ratio prior20": (
            float(row["trades"]) / prior_trades
            if pd.notna(prior_trades) and prior_trades > 0
            else float("nan")
        ),
        "1m body / range": (
            body / candle_range if candle_range > 0 else float("nan")
        ),
        "1m close location": (
            (float(row["close"]) - float(row["low"])) / candle_range
            if candle_range > 0
            else float("nan")
        ),
        "1m impulse 3m bps": impulse_3m,
        "1m realized vol prior5 bps": (
            float(realized.std() * 10_000.0)
            if len(realized) >= 3
            else float("nan")
        ),
        "1m taker buy ratio": taker_ratio,
        "1m taker buy ratio delta": (
            taker_ratio - prior_taker
            if pd.notna(taker_ratio) and pd.notna(prior_taker)
            else float("nan")
        ),
    }


def _forward_return_bps(
    one_minute: pd.DataFrame,
    entry_i: int,
    direction: int,
    horizon: int,
) -> float:
    exit_i = entry_i + int(horizon) - 1
    if exit_i >= len(one_minute):
        return float("nan")

    entry = float(one_minute.iloc[entry_i]["open"])
    exit_price = float(one_minute.iloc[exit_i]["close"])
    if entry <= 0:
        return float("nan")

    return float(direction) * (exit_price / entry - 1.0) * 10_000.0


def _merge_micro_windows(
    event_times: list[pd.Timestamp],
) -> list[dict]:
    """Merge overlapping 1m event windows to reduce Binance requests."""
    windows = []
    for event_time in sorted(pd.Timestamp(ts) for ts in event_times):
        known_time = event_time + pd.Timedelta(minutes=5)
        start = known_time - pd.Timedelta(minutes=30)
        end = known_time + pd.Timedelta(minutes=20)

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


def build_event_dataset_v3(
    universe_details: pd.DataFrame,
    *,
    forward_days: int = 7,
    warmup_hours: int = WARMUP_HOURS,
) -> pd.DataFrame:
    """Build a causal v3 event dataset with a wide context basket.

    Signal definition is deliberately broad: a new 20-bar relative breakout.
    Volume, candle shape and order-flow are recorded as features, not used to
    pre-select events. This reduces feature-selection circularity.

    Timing:
    - 5m event becomes known at its candle close
    - wait for exactly one completed 1m decision candle
    - enter on the following 1m open
    """
    if universe_details is None or universe_details.empty:
        raise ValueError("v3 universe details are missing.")

    required = {
        "Period",
        "Selection time",
        "Symbol",
        "Trade selected",
        "Context selected",
    }
    missing = required - set(universe_details.columns)
    if missing:
        raise ValueError(f"v3 universe missing columns: {sorted(missing)}")

    rows: list[dict] = []

    for period, period_frame in universe_details.groupby("Period", sort=True):
        selection_time = pd.Timestamp(period_frame["Selection time"].iloc[0])
        period_end = selection_time + pd.Timedelta(days=int(forward_days))
        fetch_start = selection_time - pd.Timedelta(hours=int(warmup_hours))

        context_symbols = (
            period_frame[period_frame["Context selected"]]
            ["Symbol"]
            .astype(str)
            .tolist()
        )
        trade_symbols = set(
            period_frame[period_frame["Trade selected"]]
            ["Symbol"]
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
            continue

        for symbol in sorted(trade_symbols):
            coin = prepared.get(symbol)
            if coin is None:
                continue

            ctx = _context_for_trade_symbol(symbol, prepared)
            if ctx.empty:
                continue

            events = _candidate_breakouts(
                coin,
                ctx,
                selection_time=selection_time,
                period_end=period_end,
            )
            events = _cooldown_per_symbol(events, minutes=30)
            if events.empty:
                continue

            event_lookup = {
                pd.Timestamp(ts): row
                for ts, row in events.iterrows()
            }

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
                    event = event_lookup[event_time]
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

                    # The bar at decision_i must complete before its features are known.
                    entry_i = decision_i + 1
                    if entry_i >= len(one_minute):
                        continue

                    features_1m = _one_minute_features(one_minute, decision_i)

                    five_row = coin.loc[event_time]
                    candle_range = float(five_row["high"]) - float(five_row["low"])
                    close_location_5m = (
                        (float(five_row["close"]) - float(five_row["low"]))
                        / candle_range
                        if candle_range > 0
                        else float("nan")
                    )

                    record = {
                        "Period": int(period),
                        "Selection time": selection_time,
                        "Symbol": symbol,
                        "Event time": event_time,
                        "Known time": known_time,
                        "Entry time": one_minute.index[entry_i],
                        "Side": "LONG" if int(event["direction"]) == 1 else "SHORT",
                        "Direction": int(event["direction"]),
                        "Breadth": float(event["breadth"]),
                        "Basket 15m return %": float(event["basket_ret_15m"]) * 100.0,
                        "Basket 60m return %": float(event["basket_ret_60m"]) * 100.0,
                        "Basket volume ratio": float(event["basket_volume_ratio"]),
                        "5m volume ratio prior20": float(event["coin_volume_ratio"]),
                        "5m trades ratio prior20": float(
                            event.get("coin_trades_ratio", float("nan"))
                        ),
                        "5m range / ATR": float(event["range_atr"]),
                        "5m relative move / ATR": float(event["relative_move_atr"]),
                        "5m ATR %": float(event["atr_pct"]) * 100.0,
                        "5m RSI": float(event["rsi"]),
                        "5m close location": close_location_5m,
                    }

                    if "5m taker buy ratio" in event:
                        record["5m taker buy ratio"] = float(
                            event["5m taker buy ratio"]
                        )
                        record["5m taker buy ratio delta"] = float(
                            event["5m taker buy ratio delta"]
                        )

                    record.update(features_1m)

                    for horizon in FORWARD_HORIZONS:
                        record[f"Gross {horizon}m bps"] = _forward_return_bps(
                            one_minute,
                            entry_i,
                            int(event["direction"]),
                            horizon,
                        )

                    rows.append(record)

    return pd.DataFrame(rows)
