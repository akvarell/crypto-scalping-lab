from __future__ import annotations

import time
from functools import lru_cache

import pandas as pd
import requests

from src.alt_basket_study import _alt_context, _cooldown, _events, _prepare_coin


BASE_URLS = [
    "https://data-api.binance.vision",
    "https://api.binance.com",
    "https://api1.binance.com",
]

HORIZONS_MINUTES = [1, 3, 5, 10, 15, 30]


def _get(path: str, params: dict, timeout: int = 20):
    last_error = None
    for base in BASE_URLS:
        try:
            response = requests.get(
                f"{base}{path}",
                params=params,
                timeout=timeout,
                headers={"User-Agent": "crypto-scalping-lab/2.0"},
            )
            response.raise_for_status()
            return response.json()
        except Exception as exc:
            last_error = exc
    raise RuntimeError(f"Binance 1m data unavailable: {last_error}")


@lru_cache(maxsize=512)
def _fetch_1m_cached(
    symbol: str,
    start_iso: str,
    end_iso: str,
) -> pd.DataFrame:
    start = pd.Timestamp(start_iso)
    end = pd.Timestamp(end_iso)
    rows: list[list] = []
    cursor = int(start.timestamp() * 1000)
    end_ms = int(end.timestamp() * 1000)
    step_ms = 60_000

    while cursor < end_ms:
        batch = _get(
            "/api/v3/klines",
            {
                "symbol": symbol,
                "interval": "1m",
                "startTime": cursor,
                "endTime": end_ms,
                "limit": 1000,
            },
        )
        if not isinstance(batch, list) or not batch:
            break

        rows.extend(batch)
        last_open = int(batch[-1][0])
        next_cursor = last_open + step_ms
        if next_cursor <= cursor:
            break
        cursor = next_cursor

        if len(batch) < 1000:
            break
        time.sleep(0.02)

    if not rows:
        return pd.DataFrame()

    df = pd.DataFrame(
        rows,
        columns=[
            "open_time", "open", "high", "low", "close", "volume",
            "close_time", "quote_volume", "trades", "taker_base",
            "taker_quote", "ignore",
        ],
    )
    df["open_time"] = pd.to_datetime(df["open_time"], unit="ms", utc=True)
    for col in ["open", "high", "low", "close", "volume", "quote_volume", "trades", "taker_base", "taker_quote"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    return (
        df.drop_duplicates(subset=["open_time"])
        .sort_values("open_time")
        .set_index("open_time")
    )


def _fetch_1m(symbol: str, start: pd.Timestamp, end: pd.Timestamp) -> pd.DataFrame:
    return _fetch_1m_cached(symbol, start.isoformat(), end.isoformat()).copy()


def _resample_5m(one_minute: pd.DataFrame) -> pd.DataFrame:
    if one_minute.empty:
        return pd.DataFrame()

    agg = {
        "open": "first",
        "high": "max",
        "low": "min",
        "close": "last",
        "volume": "sum",
        "quote_volume": "sum",
        "trades": "sum",
    }
    out = one_minute.resample("5min", label="left", closed="left").agg(agg)
    return out.dropna(subset=["open", "high", "low", "close"])


def _add_1m_micro_features(one_minute: pd.DataFrame) -> pd.DataFrame:
    out = one_minute.copy()
    vol_avg = out["volume"].rolling(20, min_periods=10).mean()
    out["volume_ratio_1m"] = out["volume"] / vol_avg.replace(0.0, float("nan"))
    out["direction_1m"] = 0
    out.loc[out["close"] > out["open"], "direction_1m"] = 1
    out.loc[out["close"] < out["open"], "direction_1m"] = -1
    return out


def _entry_index_immediate(one_minute: pd.DataFrame, known_time: pd.Timestamp) -> int | None:
    positions = one_minute.index.searchsorted(known_time, side="left")
    if positions >= len(one_minute):
        return None
    return int(positions)


def _entry_index_confirmed(
    one_minute: pd.DataFrame,
    known_time: pd.Timestamp,
    direction: int,
    confirm_window_minutes: int = 3,
) -> int | None:
    start = one_minute.index.searchsorted(known_time, side="left")
    if start >= len(one_minute):
        return None

    end = min(start + int(confirm_window_minutes), len(one_minute))
    for i in range(start, end):
        row = one_minute.iloc[i]
        if (
            int(row["direction_1m"]) == int(direction)
            and float(row["volume_ratio_1m"]) >= 1.5
        ):
            entry_i = i + 1
            return entry_i if entry_i < len(one_minute) else None

    return None


def _path_stats(
    one_minute: pd.DataFrame,
    entry_i: int,
    horizon_minutes: int,
    direction: int,
) -> dict | None:
    exit_i = entry_i + int(horizon_minutes) - 1
    if exit_i >= len(one_minute):
        return None

    entry = float(one_minute.iloc[entry_i]["open"])
    if entry <= 0:
        return None

    exit_price = float(one_minute.iloc[exit_i]["close"])
    gross_bps = int(direction) * (exit_price / entry - 1.0) * 10_000.0

    path = one_minute.iloc[entry_i : exit_i + 1]
    high = float(path["high"].max())
    low = float(path["low"].min())

    if int(direction) == 1:
        mfe_bps = (high / entry - 1.0) * 10_000.0
        mae_bps = (low / entry - 1.0) * 10_000.0
    else:
        mfe_bps = (1.0 - low / entry) * 10_000.0
        mae_bps = (1.0 - high / entry) * 10_000.0

    return {
        "entry_time": one_minute.index[entry_i],
        "exit_time": one_minute.index[exit_i],
        "gross_bps": gross_bps,
        "mfe_bps": mfe_bps,
        "mae_bps": mae_bps,
    }


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


def run_scalping_edge_map(
    rolling_details: pd.DataFrame,
    *,
    forward_days: int,
    fee_bps: float,
    slippage_bps: float,
    max_periods: int = 4,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """5m alt-basket context with 1m execution/measurement.

    A 5m event is only known after its 5m candle closes. Immediate mode enters
    at the next 1m open. Confirmed mode waits up to 3 completed 1m candles for
    same-direction + volume confirmation and then enters at the following 1m open.
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

        one_minute_data: dict[str, pd.DataFrame] = {}
        five_minute_prepared: dict[str, pd.DataFrame] = {}

        for symbol in period_sel["Symbol"].astype(str).tolist():
            try:
                one_minute = _fetch_1m(symbol, selection_time, end)
            except Exception:
                one_minute = pd.DataFrame()

            if one_minute.empty or len(one_minute) < 1000:
                continue

            one_minute = _add_1m_micro_features(one_minute)
            five_minute = _resample_5m(one_minute)
            if len(five_minute) < 200:
                continue

            one_minute_data[symbol] = one_minute
            five_minute_prepared[symbol] = _prepare_coin(five_minute)

        if len(five_minute_prepared) < 3:
            continue

        contexts = _alt_context(five_minute_prepared)

        for symbol, five_minute in five_minute_prepared.items():
            one_minute = one_minute_data[symbol]
            ctx = contexts.get(symbol)
            if ctx is None:
                continue

            events = _cooldown(_events(five_minute, ctx), minutes=60)
            if events.empty:
                continue

            for event_time, event in events.iterrows():
                direction = int(event["direction"])
                known_time = pd.Timestamp(event_time) + pd.Timedelta(minutes=5)

                entry_modes = {
                    "Immediate next 1m": _entry_index_immediate(
                        one_minute,
                        known_time,
                    ),
                    "1m volume confirm": _entry_index_confirmed(
                        one_minute,
                        known_time,
                        direction,
                        confirm_window_minutes=3,
                    ),
                }

                for entry_mode, entry_i in entry_modes.items():
                    if entry_i is None:
                        continue

                    for horizon in HORIZONS_MINUTES:
                        stats = _path_stats(
                            one_minute,
                            entry_i,
                            horizon,
                            direction,
                        )
                        if stats is None:
                            continue

                        rows.append(
                            {
                                "Period": int(period),
                                "Selection time": selection_time,
                                "Symbol": symbol,
                                "5m event time": event_time,
                                "1m entry time": stats["entry_time"],
                                "Entry mode": entry_mode,
                                "Side": "LONG" if direction == 1 else "SHORT",
                                "Alt regime": str(event["alt_regime"]),
                                "Horizon": f"{horizon}m",
                                "Gross bps": float(stats["gross_bps"]),
                                "Net bps": float(stats["gross_bps"]) - cost_bps,
                                "MFE bps": float(stats["mfe_bps"]),
                                "MAE bps": float(stats["mae_bps"]),
                                "Breadth": float(event["breadth"]),
                                "Coin volume ratio 5m": float(event["coin_volume_ratio"]),
                                "Range / ATR 5m": float(event["range_atr"]),
                                "Relative move / ATR 5m": float(event["relative_move_atr"]),
                            }
                        )

    details = pd.DataFrame(rows)
    if details.empty:
        raise RuntimeError("No 1m edge-map events were available.")

    summary_rows = []
    grouping = ["Entry mode", "Side", "Alt regime", "Horizon"]
    for keys, group in details.groupby(grouping, sort=False):
        entry_mode, side, regime, horizon = keys
        gross = group["Gross bps"].astype(float)
        net = group["Net bps"].astype(float)
        period_avg = group.groupby("Period")["Net bps"].mean()

        summary_rows.append(
            {
                "Entry mode": entry_mode,
                "Side": side,
                "Alt regime": regime,
                "Horizon": horizon,
                "Events": int(len(group)),
                "Periods": int(group["Period"].nunique()),
                "Gross avg bps": float(gross.mean()),
                "Gross median bps": float(gross.median()),
                "Trimmed gross avg bps": _trimmed_mean(gross),
                "Net avg bps": float(net.mean()),
                "Net positive events %": float((net > 0).mean() * 100.0),
                "Positive period avg": (
                    f"{int((period_avg > 0).sum())}/{len(period_avg)}"
                    if len(period_avg)
                    else "0/0"
                ),
                "Avg MFE bps": float(group["MFE bps"].mean()),
                "Avg MAE bps": float(group["MAE bps"].mean()),
                "Round-trip cost bps": cost_bps,
                "Gross edge / cost": (
                    float(gross.mean()) / cost_bps
                    if cost_bps > 0
                    else float("inf")
                ),
            }
        )

    return pd.DataFrame(summary_rows), details
