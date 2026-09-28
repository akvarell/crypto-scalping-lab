from __future__ import annotations

import pandas as pd

from src.dynamic_scalping import _fetch_5m
from src.indicators import add_indicators


HORIZONS = {"15m": 3, "30m": 6, "60m": 12}


def _prepare_coin(coin: pd.DataFrame) -> pd.DataFrame:
    out = add_indicators(
        coin,
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
    out["range_atr"] = (out["high"] - out["low"]) / out["atr"].replace(0.0, float("nan"))
    out["above_trend"] = out["close"] > out["trend_ema"]
    out["prior_high_20"] = out["high"].rolling(20, min_periods=20).max().shift(1)
    out["prior_low_20"] = out["low"].rolling(20, min_periods=20).min().shift(1)
    return out


def _alt_context(prepared: dict[str, pd.DataFrame]) -> dict[str, pd.DataFrame]:
    symbols = list(prepared)
    contexts: dict[str, pd.DataFrame] = {}

    ret15 = pd.concat(
        {s: prepared[s]["ret_15m"] for s in symbols},
        axis=1,
    )
    ret60 = pd.concat(
        {s: prepared[s]["ret_60m"] for s in symbols},
        axis=1,
    )
    breadth = pd.concat(
        {s: prepared[s]["above_trend"].astype(float) for s in symbols},
        axis=1,
    )
    volratio = pd.concat(
        {s: prepared[s]["volume_ratio"] for s in symbols},
        axis=1,
    )

    for symbol in symbols:
        others = [s for s in symbols if s != symbol]
        if not others:
            continue

        ctx = pd.DataFrame(index=prepared[symbol].index)
        ctx["basket_ret_15m"] = ret15[others].median(axis=1).reindex(ctx.index)
        ctx["basket_ret_60m"] = ret60[others].median(axis=1).reindex(ctx.index)
        ctx["breadth"] = breadth[others].mean(axis=1).reindex(ctx.index)
        ctx["basket_volume_ratio"] = volratio[others].median(axis=1).reindex(ctx.index)
        contexts[symbol] = ctx

    return contexts


def _alt_regime(ctx: pd.DataFrame) -> pd.Series:
    regime = pd.Series("NEUTRAL", index=ctx.index, dtype="object")

    bull = (
        (ctx["breadth"] >= 0.70)
        & (ctx["basket_ret_60m"] >= 0.002)
    )
    bear = (
        (ctx["breadth"] <= 0.30)
        & (ctx["basket_ret_60m"] <= -0.002)
    )

    regime[bull] = "BULL"
    regime[bear] = "BEAR"
    return regime


def _events(coin: pd.DataFrame, ctx: pd.DataFrame) -> pd.DataFrame:
    aligned = coin.join(ctx, how="left")
    aligned["rel_ret_15m"] = aligned["ret_15m"] - aligned["basket_ret_15m"]
    rel_atr = (
        aligned["rel_ret_15m"].abs()
        / aligned["atr_pct"].replace(0.0, float("nan"))
    )

    active = (
        (aligned["volume_ratio"] >= 2.0)
        & (aligned["range_atr"] >= 1.0)
        & (rel_atr >= 0.8)
    )

    long_event = (
        active
        & (aligned["close"] > aligned["prior_high_20"])
        & (aligned["close"].shift(1) <= aligned["prior_high_20"].shift(1))
        & (aligned["rel_ret_15m"] > 0)
    )
    short_event = (
        active
        & (aligned["close"] < aligned["prior_low_20"])
        & (aligned["close"].shift(1) >= aligned["prior_low_20"].shift(1))
        & (aligned["rel_ret_15m"] < 0)
    )

    out = pd.DataFrame(index=aligned.index)
    out["direction"] = 0
    out.loc[long_event, "direction"] = 1
    out.loc[short_event, "direction"] = -1
    out["breadth"] = aligned["breadth"]
    out["basket_ret_60m"] = aligned["basket_ret_60m"]
    out["basket_volume_ratio"] = aligned["basket_volume_ratio"]
    out["coin_volume_ratio"] = aligned["volume_ratio"]
    out["range_atr"] = aligned["range_atr"]
    out["relative_move_atr"] = rel_atr
    out["alt_regime"] = _alt_regime(ctx).reindex(out.index)
    return out[out["direction"] != 0].copy()


def _cooldown(events: pd.DataFrame, minutes: int = 60) -> pd.DataFrame:
    if events.empty:
        return events
    kept = []
    last_time = None
    for ts, _ in events.iterrows():
        if last_time is None or (ts - last_time) >= pd.Timedelta(minutes=minutes):
            kept.append(ts)
            last_time = ts
    return events.loc[kept]


def run_alt_basket_study(
    rolling_details: pd.DataFrame,
    *,
    forward_days: int,
    fee_bps: float,
    slippage_bps: float,
    max_periods: int = 12,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Study breakout events relative to the selected alt basket, not BTC.

    For each period, context for a coin is built from the other selected coins,
    reducing self-influence in breadth and basket-return calculations.
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

        prepared: dict[str, pd.DataFrame] = {}
        for symbol in period_sel["Symbol"].astype(str).tolist():
            try:
                market = _fetch_5m(symbol, selection_time, end)
            except Exception:
                market = pd.DataFrame()

            if market.empty or len(market) < 350:
                continue
            prepared[symbol] = _prepare_coin(market)

        if len(prepared) < 3:
            continue

        contexts = _alt_context(prepared)

        for symbol, coin in prepared.items():
            ctx = contexts.get(symbol)
            if ctx is None:
                continue

            events = _cooldown(_events(coin, ctx), minutes=60)
            if events.empty:
                continue

            pos = {ts: i for i, ts in enumerate(coin.index)}
            for ts, event in events.iterrows():
                i = pos.get(ts)
                if i is None or i + 1 >= len(coin):
                    continue

                entry_i = i + 1
                entry = float(coin.iloc[entry_i]["open"])
                if entry <= 0:
                    continue

                direction = int(event["direction"])

                for horizon, bars in HORIZONS.items():
                    exit_i = entry_i + int(bars) - 1
                    if exit_i >= len(coin):
                        continue

                    exit_price = float(coin.iloc[exit_i]["close"])
                    gross_bps = direction * (exit_price / entry - 1.0) * 10_000.0

                    rows.append(
                        {
                            "Period": int(period),
                            "Selection time": selection_time,
                            "Symbol": symbol,
                            "Event time": ts,
                            "Side": "LONG" if direction == 1 else "SHORT",
                            "Alt regime": str(event["alt_regime"]),
                            "Horizon": horizon,
                            "Gross bps": gross_bps,
                            "Net bps": gross_bps - cost_bps,
                            "Breadth": float(event["breadth"]),
                            "Basket 60m return %": float(event["basket_ret_60m"]) * 100.0,
                            "Basket volume ratio": float(event["basket_volume_ratio"]),
                            "Coin volume ratio": float(event["coin_volume_ratio"]),
                            "Range / ATR": float(event["range_atr"]),
                            "Relative move / ATR": float(event["relative_move_atr"]),
                        }
                    )

    details = pd.DataFrame(rows)
    if details.empty:
        raise RuntimeError("No alt-basket breakout events were available.")

    summary_rows = []
    for (side, regime, horizon), group in details.groupby(
        ["Side", "Alt regime", "Horizon"],
        sort=False,
    ):
        gross = group["Gross bps"].astype(float)
        net = group["Net bps"].astype(float)
        period_avg = group.groupby("Period")["Net bps"].mean()

        summary_rows.append(
            {
                "Side": side,
                "Alt regime": regime,
                "Horizon": horizon,
                "Events": int(len(group)),
                "Periods": int(group["Period"].nunique()),
                "Gross avg bps": float(gross.mean()),
                "Gross median bps": float(gross.median()),
                "Net avg bps": float(net.mean()),
                "Net positive events %": float((net > 0).mean() * 100.0),
                "Positive period avg": (
                    f"{int((period_avg > 0).sum())}/{len(period_avg)}"
                    if len(period_avg)
                    else "0/0"
                ),
                "Round-trip cost bps": cost_bps,
                "Gross edge / cost": (
                    float(gross.mean()) / cost_bps
                    if cost_bps > 0
                    else float("inf")
                ),
            }
        )

    return pd.DataFrame(summary_rows), details
