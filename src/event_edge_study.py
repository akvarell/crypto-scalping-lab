from __future__ import annotations

import pandas as pd

from src.dynamic_scalping import _fetch_5m
from src.relative_event_lab import _prepare_relative_features


TIERS = [
    {
        "name": "Base breakout",
        "max_abs_corr": 0.45,
        "min_volume_ratio": 2.0,
        "min_range_atr": 1.0,
        "min_relative_atr": 0.8,
    },
    {
        "name": "Strong breakout",
        "max_abs_corr": 0.35,
        "min_volume_ratio": 3.0,
        "min_range_atr": 1.5,
        "min_relative_atr": 1.1,
    },
    {
        "name": "Extreme breakout",
        "max_abs_corr": 0.25,
        "min_volume_ratio": 4.0,
        "min_range_atr": 2.0,
        "min_relative_atr": 1.4,
    },
]

HORIZONS = {
    "15m": 3,
    "30m": 6,
    "60m": 12,
}


def _events_for_tier(features: pd.DataFrame, tier: dict) -> pd.DataFrame:
    out = features.copy()

    rel_atr = (
        out["rel_ret_3"].abs()
        / out["atr_pct"].replace(0.0, float("nan"))
    )

    low_corr = out["btc_corr"].abs() <= float(tier["max_abs_corr"])
    active = (
        low_corr
        & (out["volume_ratio"] >= float(tier["min_volume_ratio"]))
        & (out["range_atr"] >= float(tier["min_range_atr"]))
        & (rel_atr >= float(tier["min_relative_atr"]))
    )

    long_event = (
        active
        & (out["close"] > out["prior_high_20"])
        & (out["close"].shift(1) <= out["prior_high_20"].shift(1))
        & (out["rel_ret_3"] > 0)
    )
    short_event = (
        active
        & (out["close"] < out["prior_low_20"])
        & (out["close"].shift(1) >= out["prior_low_20"].shift(1))
        & (out["rel_ret_3"] < 0)
    )

    events = pd.DataFrame(index=out.index)
    events["direction"] = 0
    events.loc[long_event, "direction"] = 1
    events.loc[short_event, "direction"] = -1
    events["volume_ratio"] = out["volume_ratio"]
    events["range_atr"] = out["range_atr"]
    events["btc_corr"] = out["btc_corr"]
    events["rel_atr"] = rel_atr
    return events[events["direction"] != 0].copy()


def _cooldown_filter(events: pd.DataFrame, bars: int = 12) -> pd.DataFrame:
    if events.empty:
        return events

    kept = []
    last_pos = -10_000
    positions = {ts: i for i, ts in enumerate(events.index)}

    # Events index is sparse; cooldown is enforced by elapsed clock time.
    last_time = None
    for ts, row in events.iterrows():
        if last_time is None or (ts - last_time) >= pd.Timedelta(minutes=5 * bars):
            kept.append(ts)
            last_time = ts

    return events.loc[kept].copy()


def run_event_edge_study(
    rolling_details: pd.DataFrame,
    *,
    forward_days: int,
    fee_bps: float,
    slippage_bps: float,
    max_periods: int = 12,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Measure fixed-horizon edge after BTC-relative breakout events.

    Entry is the next 5m open. Exit is a fixed future 5m close. A 60-minute
    per-symbol event cooldown reduces repeated counting of the same impulse.
    """
    if rolling_details is None or rolling_details.empty:
        raise ValueError("Run Rolling Universe Validation first.")

    selected = rolling_details[["Period", "Selection time", "Symbol"]].drop_duplicates().copy()
    selected["Selection time"] = pd.to_datetime(selected["Selection time"], utc=True)
    keep_periods = sorted(selected["Period"].unique())[-int(max_periods):]
    selected = selected[selected["Period"].isin(keep_periods)]

    round_trip_cost_bps = 2.0 * (float(fee_bps) + float(slippage_bps))

    market_cache: dict[tuple[str, pd.Timestamp], pd.DataFrame] = {}
    btc_cache: dict[pd.Timestamp, pd.DataFrame] = {}
    event_rows = []

    for _, sel in selected.iterrows():
        period = int(sel["Period"])
        selection_time = pd.Timestamp(sel["Selection time"])
        symbol = str(sel["Symbol"])
        forward_end = selection_time + pd.Timedelta(days=int(forward_days))
        key = (symbol, selection_time)

        if selection_time not in btc_cache:
            try:
                btc_cache[selection_time] = _fetch_5m(
                    "BTCUSDT",
                    selection_time,
                    forward_end,
                )
            except Exception:
                btc_cache[selection_time] = pd.DataFrame()

        if key not in market_cache:
            try:
                market_cache[key] = _fetch_5m(
                    symbol,
                    selection_time,
                    forward_end,
                )
            except Exception:
                market_cache[key] = pd.DataFrame()

        coin = market_cache[key]
        btc = btc_cache[selection_time]
        if coin.empty or btc.empty or len(coin) < 200 or len(btc) < 200:
            continue

        features = _prepare_relative_features(coin, btc)
        index_lookup = {ts: i for i, ts in enumerate(features.index)}

        for tier in TIERS:
            events = _cooldown_filter(_events_for_tier(features, tier), bars=12)
            if events.empty:
                continue

            for ts, event in events.iterrows():
                i = index_lookup.get(ts)
                if i is None or i + 1 >= len(features):
                    continue

                entry_i = i + 1
                entry_price = float(features.iloc[entry_i]["open"])
                if entry_price <= 0:
                    continue

                direction = int(event["direction"])

                for horizon_name, bars in HORIZONS.items():
                    exit_i = entry_i + int(bars) - 1
                    if exit_i >= len(features):
                        continue

                    exit_price = float(features.iloc[exit_i]["close"])
                    gross_bps = direction * (exit_price / entry_price - 1.0) * 10_000.0
                    net_bps = gross_bps - round_trip_cost_bps

                    event_rows.append(
                        {
                            "Tier": tier["name"],
                            "Horizon": horizon_name,
                            "Period": period,
                            "Selection time": selection_time,
                            "Symbol": symbol,
                            "Event time": ts,
                            "Side": "LONG" if direction == 1 else "SHORT",
                            "Gross bps": gross_bps,
                            "Net bps": net_bps,
                            "Volume ratio": float(event["volume_ratio"]),
                            "Range / ATR": float(event["range_atr"]),
                            "BTC corr": float(event["btc_corr"]),
                            "Relative move / ATR": float(event["rel_atr"]),
                        }
                    )

    details = pd.DataFrame(event_rows)
    if details.empty:
        raise RuntimeError("No breakout events matched the fixed event-quality tiers.")

    summary_rows = []
    for (tier, horizon), group in details.groupby(["Tier", "Horizon"], sort=False):
        gross = group["Gross bps"].astype(float)
        net = group["Net bps"].astype(float)

        period_means = group.groupby("Period")["Net bps"].mean()
        long_group = group[group["Side"] == "LONG"]
        short_group = group[group["Side"] == "SHORT"]

        summary_rows.append(
            {
                "Tier": tier,
                "Horizon": horizon,
                "Events": int(len(group)),
                "Periods with events": int(group["Period"].nunique()),
                "Gross avg bps": float(gross.mean()),
                "Gross median bps": float(gross.median()),
                "Net avg bps": float(net.mean()),
                "Net positive events %": float((net > 0).mean() * 100.0),
                "Positive period avg": (
                    f"{int((period_means > 0).sum())}/{len(period_means)}"
                    if len(period_means)
                    else "0/0"
                ),
                "LONG gross avg bps": (
                    float(long_group["Gross bps"].mean())
                    if not long_group.empty
                    else float("nan")
                ),
                "SHORT gross avg bps": (
                    float(short_group["Gross bps"].mean())
                    if not short_group.empty
                    else float("nan")
                ),
                "Round-trip cost bps": round_trip_cost_bps,
                "Gross edge / cost": (
                    float(gross.mean()) / round_trip_cost_bps
                    if round_trip_cost_bps > 0
                    else float("inf")
                ),
            }
        )

    summary = pd.DataFrame(summary_rows)
    return summary, details
