from __future__ import annotations

import pandas as pd

from src.dynamic_scalping import _fetch_5m
from src.relative_event_lab import _prepare_relative_features


HORIZONS = {"15m": 3, "30m": 6, "60m": 12}


def _btc_regime(btc: pd.DataFrame) -> pd.Series:
    close = btc["close"].astype(float)
    ret_24h = close.pct_change(288)
    ema_fast = close.ewm(span=48, adjust=False).mean()
    ema_slow = close.ewm(span=192, adjust=False).mean()

    regime = pd.Series("NEUTRAL", index=btc.index, dtype="object")
    regime[(ret_24h >= 0.01) & (ema_fast > ema_slow)] = "BULL"
    regime[(ret_24h <= -0.01) & (ema_fast < ema_slow)] = "BEAR"
    return regime


def _base_breakout_events(features: pd.DataFrame) -> pd.DataFrame:
    rel_atr = features["rel_ret_3"].abs() / features["atr_pct"].replace(0.0, float("nan"))
    active = (
        (features["btc_corr"].abs() <= 0.45)
        & (features["volume_ratio"] >= 2.0)
        & (features["range_atr"] >= 1.0)
        & (rel_atr >= 0.8)
    )

    long_event = (
        active
        & (features["close"] > features["prior_high_20"])
        & (features["close"].shift(1) <= features["prior_high_20"].shift(1))
        & (features["rel_ret_3"] > 0)
    )
    short_event = (
        active
        & (features["close"] < features["prior_low_20"])
        & (features["close"].shift(1) >= features["prior_low_20"].shift(1))
        & (features["rel_ret_3"] < 0)
    )

    out = pd.DataFrame(index=features.index)
    out["direction"] = 0
    out.loc[long_event, "direction"] = 1
    out.loc[short_event, "direction"] = -1
    out["btc_corr"] = features["btc_corr"]
    out["volume_ratio"] = features["volume_ratio"]
    out["range_atr"] = features["range_atr"]
    out["rel_atr"] = rel_atr
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


def run_direction_regime_study(
    rolling_details: pd.DataFrame,
    *,
    forward_days: int,
    fee_bps: float,
    slippage_bps: float,
    max_periods: int = 12,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    if rolling_details is None or rolling_details.empty:
        raise ValueError("Run Rolling Universe Validation first.")

    selected = rolling_details[["Period", "Selection time", "Symbol"]].drop_duplicates().copy()
    selected["Selection time"] = pd.to_datetime(selected["Selection time"], utc=True)
    keep_periods = sorted(selected["Period"].unique())[-int(max_periods):]
    selected = selected[selected["Period"].isin(keep_periods)]

    cost_bps = 2.0 * (float(fee_bps) + float(slippage_bps))
    rows = []
    market_cache: dict[tuple[str, pd.Timestamp], pd.DataFrame] = {}
    btc_cache: dict[pd.Timestamp, pd.DataFrame] = {}

    for _, sel in selected.iterrows():
        period = int(sel["Period"])
        selection_time = pd.Timestamp(sel["Selection time"])
        symbol = str(sel["Symbol"])
        end = selection_time + pd.Timedelta(days=int(forward_days))
        key = (symbol, selection_time)

        if selection_time not in btc_cache:
            try:
                btc_cache[selection_time] = _fetch_5m("BTCUSDT", selection_time, end)
            except Exception:
                btc_cache[selection_time] = pd.DataFrame()

        if key not in market_cache:
            try:
                market_cache[key] = _fetch_5m(symbol, selection_time, end)
            except Exception:
                market_cache[key] = pd.DataFrame()

        coin = market_cache[key]
        btc = btc_cache[selection_time]
        if coin.empty or btc.empty or len(coin) < 350 or len(btc) < 350:
            continue

        features = _prepare_relative_features(coin, btc)
        btc_regime = _btc_regime(btc).reindex(features.index).ffill()
        events = _cooldown(_base_breakout_events(features), minutes=60)
        pos = {ts: i for i, ts in enumerate(features.index)}

        for ts, event in events.iterrows():
            i = pos.get(ts)
            if i is None or i + 1 >= len(features):
                continue
            entry_i = i + 1
            entry = float(features.iloc[entry_i]["open"])
            if entry <= 0:
                continue

            direction = int(event["direction"])
            regime = str(btc_regime.loc[ts]) if ts in btc_regime.index else "NEUTRAL"

            for horizon, bars in HORIZONS.items():
                exit_i = entry_i + int(bars) - 1
                if exit_i >= len(features):
                    continue
                exit_price = float(features.iloc[exit_i]["close"])
                gross_bps = direction * (exit_price / entry - 1.0) * 10_000.0

                rows.append(
                    {
                        "Period": period,
                        "Symbol": symbol,
                        "Event time": ts,
                        "Side": "LONG" if direction == 1 else "SHORT",
                        "BTC regime": regime,
                        "Horizon": horizon,
                        "Gross bps": gross_bps,
                        "Net bps": gross_bps - cost_bps,
                        "BTC corr": float(event["btc_corr"]),
                        "Volume ratio": float(event["volume_ratio"]),
                        "Range / ATR": float(event["range_atr"]),
                        "Relative move / ATR": float(event["rel_atr"]),
                    }
                )

    details = pd.DataFrame(rows)
    if details.empty:
        raise RuntimeError("No regime-study events were available.")

    summary_rows = []
    for (side, regime, horizon), group in details.groupby(
        ["Side", "BTC regime", "Horizon"], sort=False
    ):
        gross = group["Gross bps"].astype(float)
        net = group["Net bps"].astype(float)
        period_avg = group.groupby("Period")["Net bps"].mean()

        summary_rows.append(
            {
                "Side": side,
                "BTC regime": regime,
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

    summary = pd.DataFrame(summary_rows)
    return summary, details
