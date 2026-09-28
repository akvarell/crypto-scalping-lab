from __future__ import annotations

import pandas as pd

from src.alt_basket_study import _alt_context, _cooldown, _events, _prepare_coin
from src.dynamic_scalping import _fetch_5m
from src.rolling_universe import run_rolling_universe_validation


RULES = [
    {
        "name": "LONG in alt-BEAR · 30m",
        "side": "LONG",
        "regime": "BEAR",
        "bars": 6,
    },
    {
        "name": "SHORT in alt-BULL · 15m",
        "side": "SHORT",
        "regime": "BULL",
        "bars": 3,
    },
]


def _trimmed_mean(series: pd.Series, lower: float = 0.10, upper: float = 0.90) -> float:
    clean = series.dropna().astype(float)
    if clean.empty:
        return 0.0
    if len(clean) < 10:
        return float(clean.mean())
    lo = clean.quantile(lower)
    hi = clean.quantile(upper)
    trimmed = clean[(clean >= lo) & (clean <= hi)]
    return float(trimmed.mean()) if not trimmed.empty else float(clean.mean())


def _evaluate_rules(
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

            for rule in RULES:
                direction = 1 if rule["side"] == "LONG" else -1
                subset = events[
                    (events["direction"] == direction)
                    & (events["alt_regime"] == rule["regime"])
                ]
                if subset.empty:
                    continue

                for ts, event in subset.iterrows():
                    i = pos.get(ts)
                    if i is None or i + 1 >= len(coin):
                        continue

                    entry_i = i + 1
                    exit_i = entry_i + int(rule["bars"]) - 1
                    if exit_i >= len(coin):
                        continue

                    entry = float(coin.iloc[entry_i]["open"])
                    exit_price = float(coin.iloc[exit_i]["close"])
                    if entry <= 0:
                        continue

                    gross_bps = direction * (exit_price / entry - 1.0) * 10_000.0
                    rows.append(
                        {
                            "Rule": rule["name"],
                            "Period": int(period),
                            "Selection time": selection_time,
                            "Symbol": symbol,
                            "Event time": ts,
                            "Side": rule["side"],
                            "Alt regime": rule["regime"],
                            "Gross bps": gross_bps,
                            "Net bps": gross_bps - cost_bps,
                            "Breadth": float(event["breadth"]),
                            "Coin volume ratio": float(event["coin_volume_ratio"]),
                            "Range / ATR": float(event["range_atr"]),
                            "Relative move / ATR": float(event["relative_move_atr"]),
                        }
                    )

    details = pd.DataFrame(rows)
    if details.empty:
        raise RuntimeError("No fixed counter-regime events appeared in the holdout window.")

    combined = details.copy()
    combined["Rule"] = "COMBINED fixed rule"
    combined_details = pd.concat([details, combined], ignore_index=True)

    summary_rows = []
    for rule, group in combined_details.groupby("Rule", sort=False):
        gross = group["Gross bps"].astype(float)
        net = group["Net bps"].astype(float)
        period_avg = group.groupby("Period")["Net bps"].mean()

        summary_rows.append(
            {
                "Rule": rule,
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

    return pd.DataFrame(summary_rows), details


def run_counter_regime_holdout(
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
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, dict]:
    """Freeze the v1.8 counter-regime observation and test an older window.

    No thresholds or horizons are optimized inside the holdout:
      - LONG in alt-BEAR, exit after 30m
      - SHORT in alt-BULL, exit after 15m
    """
    periods, rolling_details, rolling_summary = run_rolling_universe_validation(
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

    summary, event_details = _evaluate_rules(
        rolling_details,
        forward_days=int(forward_days),
        fee_bps=float(fee_bps),
        slippage_bps=float(slippage_bps),
    )
    return summary, event_details, periods, rolling_summary
