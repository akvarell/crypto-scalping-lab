from __future__ import annotations

import io
import time
import zipfile
from functools import lru_cache

import numpy as np
import pandas as pd
import requests

from src.dynamic_scalping import _fetch_5m
from src.event_dataset_v3 import _forward_return_bps
from src.microstructure_context_v33 import analyze_microstructure_context_v33
from src.scalping_edge_map import _fetch_1m


FAPI_BASES = [
    "https://fapi.binance.com",
    "https://fapi1.binance.com",
]
HORIZONS = [3, 5, 10, 15, 30]


ARCHIVE_BASE = "https://data.binance.vision/data/futures/um"


def _archive_months(start: pd.Timestamp, end: pd.Timestamp) -> list[pd.Timestamp]:
    first = pd.Timestamp(start).tz_convert("UTC").normalize().replace(day=1)
    last = pd.Timestamp(end).tz_convert("UTC").normalize().replace(day=1)
    return list(pd.date_range(first, last, freq="MS", tz="UTC"))


def _parse_kline_zip(content: bytes) -> pd.DataFrame:
    with zipfile.ZipFile(io.BytesIO(content)) as zf:
        names = [name for name in zf.namelist() if name.lower().endswith(".csv")]
        if not names:
            return pd.DataFrame()
        raw = pd.read_csv(zf.open(names[0]), header=None)

    if raw.empty or raw.shape[1] < 5:
        return pd.DataFrame()

    raw = raw.iloc[:, :12].copy()
    raw.columns = [
        "open_time", "open", "high", "low", "close", "volume",
        "close_time", "quote_volume", "trades", "taker_base",
        "taker_quote", "ignore",
    ][: raw.shape[1]]

    numeric_time = pd.to_numeric(raw["open_time"], errors="coerce")
    raw = raw.loc[numeric_time.notna()].copy()
    if raw.empty:
        return pd.DataFrame()

    numeric_time = pd.to_numeric(raw["open_time"], errors="coerce")
    unit = "us" if float(numeric_time.abs().median()) > 1e14 else "ms"
    raw["open_time"] = pd.to_datetime(numeric_time, unit=unit, utc=True)

    for col in [x for x in ["open", "high", "low", "close", "volume", "quote_volume", "trades", "taker_base", "taker_quote"] if x in raw]:
        raw[col] = pd.to_numeric(raw[col], errors="coerce")

    return (
        raw.drop_duplicates(subset=["open_time"])
        .sort_values("open_time")
        .set_index("open_time")
    )


def _archive_url(
    data_type: str,
    symbol: str,
    interval: str,
    stamp: pd.Timestamp,
    *,
    monthly: bool,
) -> str:
    symbol = symbol.upper()
    if monthly:
        label = stamp.strftime("%Y-%m")
        return (
            f"{ARCHIVE_BASE}/monthly/{data_type}/{symbol}/{interval}/"
            f"{symbol}-{interval}-{label}.zip"
        )
    label = stamp.strftime("%Y-%m-%d")
    return (
        f"{ARCHIVE_BASE}/daily/{data_type}/{symbol}/{interval}/"
        f"{symbol}-{interval}-{label}.zip"
    )


def _download_archive_frame(
    data_type: str,
    symbol: str,
    start: pd.Timestamp,
    end: pd.Timestamp,
    interval: str = "5m",
) -> pd.DataFrame:
    pieces: list[pd.DataFrame] = []
    start = pd.Timestamp(start).tz_convert("UTC")
    end = pd.Timestamp(end).tz_convert("UTC")
    current_month = pd.Timestamp.now(tz="UTC").normalize().replace(day=1)

    for month in _archive_months(start, end):
        month_end = month + pd.offsets.MonthBegin(1)
        overlap_start = max(start, month)
        overlap_end = min(end, month_end)

        month_frame = pd.DataFrame()
        if month < current_month:
            url = _archive_url(
                data_type,
                symbol,
                interval,
                month,
                monthly=True,
            )
            try:
                response = requests.get(
                    url,
                    timeout=30,
                    headers={"User-Agent": "crypto-scalping-lab/3.3.3"},
                )
                if response.status_code == 200 and response.content:
                    month_frame = _parse_kline_zip(response.content)
            except Exception:
                month_frame = pd.DataFrame()

        if not month_frame.empty:
            pieces.append(
                month_frame[
                    (month_frame.index >= overlap_start)
                    & (month_frame.index < overlap_end)
                ]
            )
            continue

        for day in pd.date_range(
            overlap_start.normalize(),
            (overlap_end - pd.Timedelta(microseconds=1)).normalize(),
            freq="D",
            tz="UTC",
        ):
            url = _archive_url(
                data_type,
                symbol,
                interval,
                day,
                monthly=False,
            )
            try:
                response = requests.get(
                    url,
                    timeout=30,
                    headers={"User-Agent": "crypto-scalping-lab/3.3.3"},
                )
                if response.status_code != 200 or not response.content:
                    continue
                daily = _parse_kline_zip(response.content)
                if not daily.empty:
                    pieces.append(daily)
            except Exception:
                continue

    if not pieces:
        return pd.DataFrame()

    frame = pd.concat(pieces).sort_index()
    frame = frame[~frame.index.duplicated(keep="last")]
    return frame[(frame.index >= start) & (frame.index < end)].copy()


def _fget(path: str, params: dict, timeout: int = 20):
    last_error = None
    for base in FAPI_BASES:
        try:
            response = requests.get(
                f"{base}{path}",
                params=params,
                timeout=timeout,
                headers={"User-Agent": "crypto-scalping-lab/3.3.2"},
            )
            response.raise_for_status()
            return response.json()
        except Exception as exc:
            last_error = exc
    raise RuntimeError(f"Binance USD-M futures data unavailable: {last_error}")


def _paginate_5m(path: str, symbol: str, start: pd.Timestamp, end: pd.Timestamp) -> list[list]:
    rows: list[list] = []
    cursor = int(pd.Timestamp(start).timestamp() * 1000)
    end_ms = int(pd.Timestamp(end).timestamp() * 1000)
    step_ms = 300_000

    while cursor < end_ms:
        batch = _fget(
            path,
            {
                "symbol": symbol,
                "interval": "5m",
                "startTime": cursor,
                "endTime": end_ms,
                "limit": 1500,
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
        if len(batch) < 1500:
            break
        time.sleep(0.02)
    return rows


@lru_cache(maxsize=256)
def _fetch_premium_5m_cached(symbol: str, start_iso: str, end_iso: str) -> pd.DataFrame:
    rows = _paginate_5m(
        "/fapi/v1/premiumIndexKlines",
        symbol,
        pd.Timestamp(start_iso),
        pd.Timestamp(end_iso),
    )
    if not rows:
        return pd.DataFrame()

    df = pd.DataFrame(
        rows,
        columns=[
            "open_time", "open", "high", "low", "close", "ignore1",
            "close_time", "ignore2", "ignore3", "ignore4", "ignore5", "ignore6",
        ],
    )
    df["open_time"] = pd.to_datetime(df["open_time"], unit="ms", utc=True)
    for col in ["open", "high", "low", "close"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    return (
        df.drop_duplicates(subset=["open_time"])
        .sort_values("open_time")
        .set_index("open_time")
    )


@lru_cache(maxsize=256)
def _fetch_futures_5m_cached(symbol: str, start_iso: str, end_iso: str) -> pd.DataFrame:
    rows = _paginate_5m(
        "/fapi/v1/klines",
        symbol,
        pd.Timestamp(start_iso),
        pd.Timestamp(end_iso),
    )
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
    for col in ["open", "high", "low", "close", "volume", "quote_volume", "trades"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    return (
        df.drop_duplicates(subset=["open_time"])
        .sort_values("open_time")
        .set_index("open_time")
    )


@lru_cache(maxsize=256)
def _fetch_funding_cached(symbol: str, start_iso: str, end_iso: str) -> pd.DataFrame:
    start = pd.Timestamp(start_iso)
    end = pd.Timestamp(end_iso)
    cursor = int(start.timestamp() * 1000)
    end_ms = int(end.timestamp() * 1000)
    rows: list[dict] = []

    while cursor <= end_ms:
        batch = _fget(
            "/fapi/v1/fundingRate",
            {
                "symbol": symbol,
                "startTime": cursor,
                "endTime": end_ms,
                "limit": 1000,
            },
        )
        if not isinstance(batch, list) or not batch:
            break
        rows.extend(batch)
        last_time = int(batch[-1]["fundingTime"])
        next_cursor = last_time + 1
        if next_cursor <= cursor:
            break
        cursor = next_cursor
        if len(batch) < 1000:
            break
        time.sleep(0.02)

    if not rows:
        return pd.DataFrame()

    df = pd.DataFrame(rows)
    df["fundingTime"] = pd.to_datetime(df["fundingTime"], unit="ms", utc=True)
    df["fundingRate"] = pd.to_numeric(df["fundingRate"], errors="coerce")
    return (
        df.drop_duplicates(subset=["fundingTime"])
        .sort_values("fundingTime")
        .set_index("fundingTime")
    )


def clear_futures_v332_caches() -> None:
    _fetch_premium_5m_cached.cache_clear()
    _fetch_futures_5m_cached.cache_clear()
    _fetch_funding_cached.cache_clear()


def _fetch_premium_5m(symbol: str, start: pd.Timestamp, end: pd.Timestamp) -> pd.DataFrame:
    try:
        frame = _fetch_premium_5m_cached(
            symbol,
            start.isoformat(),
            end.isoformat(),
        ).copy()
    except Exception:
        frame = pd.DataFrame()

    if frame.empty:
        frame = _download_archive_frame(
            "premiumIndexKlines",
            symbol,
            start,
            end,
            interval="5m",
        )
    return frame


def _fetch_futures_5m(symbol: str, start: pd.Timestamp, end: pd.Timestamp) -> pd.DataFrame:
    try:
        frame = _fetch_futures_5m_cached(
            symbol,
            start.isoformat(),
            end.isoformat(),
        ).copy()
    except Exception:
        frame = pd.DataFrame()

    if frame.empty:
        frame = _download_archive_frame(
            "klines",
            symbol,
            start,
            end,
            interval="5m",
        )
    return frame


def _fetch_funding(symbol: str, start: pd.Timestamp, end: pd.Timestamp) -> pd.DataFrame:
    return _fetch_funding_cached(symbol, start.isoformat(), end.isoformat()).copy()


def _last_known_funding(funding: pd.DataFrame, known_time: pd.Timestamp) -> tuple[float, float]:
    if funding.empty:
        return float("nan"), float("nan")

    times = funding.index
    pos = int(times.searchsorted(known_time, side="right")) - 1
    if pos < 0:
        return float("nan"), float("nan")

    current = float(funding.iloc[pos]["fundingRate"])
    previous = (
        float(funding.iloc[pos - 1]["fundingRate"])
        if pos >= 1
        else float("nan")
    )
    change = (
        current - previous
        if np.isfinite(current) and np.isfinite(previous)
        else float("nan")
    )
    return current, change


def _append_obs(
    rows: list[dict],
    *,
    period: int,
    symbol: str,
    category: str,
    feature: str,
    side_sign: int,
    strength: float,
    raw_value: float,
    future_returns: dict[int, float],
) -> None:
    if side_sign not in (-1, 1):
        return
    if not np.isfinite(strength) or float(strength) <= 0:
        return

    row = {
        "Period": int(period),
        "Symbol": symbol,
        "Category": category,
        "Feature": feature,
        "Side": "LONG" if side_sign == 1 else "SHORT",
        "Strength": float(strength),
        "Raw value": float(raw_value) if np.isfinite(raw_value) else float("nan"),
    }
    for horizon, raw_bps in future_returns.items():
        row[f"Gross {horizon}m bps"] = (
            float(side_sign) * float(raw_bps)
            if np.isfinite(raw_bps)
            else float("nan")
        )
    rows.append(row)


def build_futures_positioning_period_v332(
    period_frame: pd.DataFrame,
    *,
    forward_days: int = 7,
    warmup_hours: int = 48,
    sample_minutes: int = 60,
) -> pd.DataFrame:
    """Build threshold-free futures-positioning observations for one period."""
    if period_frame is None or period_frame.empty:
        return pd.DataFrame()

    period = int(period_frame["Period"].iloc[0])
    selection_time = pd.Timestamp(period_frame["Selection time"].iloc[0])
    period_end = selection_time + pd.Timedelta(days=int(forward_days))
    history_start = selection_time - pd.Timedelta(hours=int(warmup_hours))

    symbols = (
        period_frame[period_frame["Trade selected"]]["Symbol"]
        .astype(str)
        .tolist()
    )

    rows: list[dict] = []

    for symbol in symbols:
        try:
            spot5 = _fetch_5m(symbol, history_start, period_end)
            spot1 = _fetch_1m(
                symbol,
                selection_time - pd.Timedelta(minutes=5),
                period_end + pd.Timedelta(minutes=40),
            )
        except Exception:
            continue

        if spot5.empty or spot1.empty:
            continue

        # Futures sources are independent. A failure in funding must never
        # discard valid premium/basis observations for the same symbol.
        try:
            premium = _fetch_premium_5m(symbol, history_start, period_end)
        except Exception:
            premium = pd.DataFrame()

        try:
            futures = _fetch_futures_5m(symbol, history_start, period_end)
        except Exception:
            futures = pd.DataFrame()

        try:
            funding = _fetch_funding(
                symbol,
                history_start - pd.Timedelta(days=2),
                period_end,
            )
        except Exception:
            funding = pd.DataFrame()

        if premium.empty and futures.empty and funding.empty:
            continue

        frame = pd.DataFrame(index=spot5.index)
        frame["spot_close"] = spot5["close"]
        frame["futures_close"] = (
            futures["close"].reindex(frame.index)
            if not futures.empty and "close" in futures
            else float("nan")
        )
        frame["premium"] = (
            premium["close"].reindex(frame.index)
            if not premium.empty and "close" in premium
            else float("nan")
        )
        frame["premium_change_15m"] = frame["premium"].diff(3)
        frame["basis"] = (
            frame["futures_close"]
            / frame["spot_close"].replace(0.0, float("nan"))
            - 1.0
        )
        frame["basis_change_15m"] = frame["basis"].diff(3)

        sample_index = frame.index[
            (frame.index >= selection_time)
            & (frame.index < period_end)
            & (frame.index.minute % int(sample_minutes) == 0)
        ]

        for ts in sample_index:
            known_time = pd.Timestamp(ts) + pd.Timedelta(minutes=5)
            entry_i = int(spot1.index.searchsorted(known_time, side="left"))
            if entry_i >= len(spot1):
                continue

            raw_future = {
                horizon: _forward_return_bps(spot1, entry_i, 1, horizon)
                for horizon in HORIZONS
            }

            row = frame.loc[ts]
            premium_level = float(row.get("premium", float("nan")))
            premium_change = float(row.get("premium_change_15m", float("nan")))
            basis = float(row.get("basis", float("nan")))
            basis_change = float(row.get("basis_change_15m", float("nan")))
            funding_rate, funding_change = _last_known_funding(
                funding,
                known_time,
            )

            if np.isfinite(premium_level) and premium_level != 0:
                sign = 1 if premium_level > 0 else -1
                _append_obs(
                    rows,
                    period=period,
                    symbol=symbol,
                    category="Perpetual premium",
                    feature="Premium continuation",
                    side_sign=sign,
                    strength=abs(premium_level),
                    raw_value=premium_level,
                    future_returns=raw_future,
                )
                _append_obs(
                    rows,
                    period=period,
                    symbol=symbol,
                    category="Perpetual premium",
                    feature="Premium crowding reversion",
                    side_sign=-sign,
                    strength=abs(premium_level),
                    raw_value=premium_level,
                    future_returns=raw_future,
                )

            if np.isfinite(premium_change) and premium_change != 0:
                sign = 1 if premium_change > 0 else -1
                _append_obs(
                    rows,
                    period=period,
                    symbol=symbol,
                    category="Perpetual premium",
                    feature="Premium-change continuation",
                    side_sign=sign,
                    strength=abs(premium_change),
                    raw_value=premium_change,
                    future_returns=raw_future,
                )
                _append_obs(
                    rows,
                    period=period,
                    symbol=symbol,
                    category="Perpetual premium",
                    feature="Premium-change reversion",
                    side_sign=-sign,
                    strength=abs(premium_change),
                    raw_value=premium_change,
                    future_returns=raw_future,
                )

            if np.isfinite(funding_rate) and funding_rate != 0:
                sign = 1 if funding_rate > 0 else -1
                _append_obs(
                    rows,
                    period=period,
                    symbol=symbol,
                    category="Funding",
                    feature="Funding crowding reversion",
                    side_sign=-sign,
                    strength=abs(funding_rate),
                    raw_value=funding_rate,
                    future_returns=raw_future,
                )

            if np.isfinite(funding_change) and funding_change != 0:
                sign = 1 if funding_change > 0 else -1
                _append_obs(
                    rows,
                    period=period,
                    symbol=symbol,
                    category="Funding",
                    feature="Funding-change reversion",
                    side_sign=-sign,
                    strength=abs(funding_change),
                    raw_value=funding_change,
                    future_returns=raw_future,
                )

            if np.isfinite(basis) and basis != 0:
                sign = 1 if basis > 0 else -1
                _append_obs(
                    rows,
                    period=period,
                    symbol=symbol,
                    category="Futures vs spot",
                    feature="Basis reversion",
                    side_sign=-sign,
                    strength=abs(basis),
                    raw_value=basis,
                    future_returns=raw_future,
                )
                _append_obs(
                    rows,
                    period=period,
                    symbol=symbol,
                    category="Futures vs spot",
                    feature="Basis continuation",
                    side_sign=sign,
                    strength=abs(basis),
                    raw_value=basis,
                    future_returns=raw_future,
                )

            if np.isfinite(basis_change) and basis_change != 0:
                sign = 1 if basis_change > 0 else -1
                _append_obs(
                    rows,
                    period=period,
                    symbol=symbol,
                    category="Futures vs spot",
                    feature="Basis-change reversion",
                    side_sign=-sign,
                    strength=abs(basis_change),
                    raw_value=basis_change,
                    future_returns=raw_future,
                )

    return pd.DataFrame(rows)


def analyze_futures_positioning_v332(
    observations: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, str]:
    surface, candidates, verdict = analyze_microstructure_context_v33(observations)
    verdict = verdict.replace("v3.3", "v3.3.2")
    verdict = verdict.replace(
        "microstructure/context",
        "futures-positioning/basis",
    )
    return surface, candidates, verdict
