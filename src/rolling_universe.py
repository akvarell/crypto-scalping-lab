from __future__ import annotations

import math
import time
from datetime import datetime, timezone

import pandas as pd
import requests


BASE_URLS = [
    "https://data-api.binance.vision",
    "https://api.binance.com",
    "https://api1.binance.com",
]

EXCLUDED_BASES = {
    "USDT", "USDC", "FDUSD", "TUSD", "DAI", "USDP", "EUR", "TRY", "GBP", "BRL",
}


def _get(path: str, params: dict | None = None, timeout: int = 20):
    last_error = None
    for base in BASE_URLS:
        try:
            response = requests.get(
                f"{base}{path}",
                params=params,
                timeout=timeout,
                headers={"User-Agent": "crypto-scalping-lab/1.0"},
            )
            response.raise_for_status()
            return response.json()
        except Exception as exc:
            last_error = exc
    raise RuntimeError(f"Binance public market data unavailable: {last_error}")


def _valid_usdt_symbol(info: dict) -> bool:
    base = str(info.get("baseAsset", ""))
    quote = str(info.get("quoteAsset", ""))
    status = str(info.get("status", ""))
    symbol = str(info.get("symbol", ""))

    if quote != "USDT" or status != "TRADING":
        return False
    if base in EXCLUDED_BASES:
        return False
    if any(base.endswith(s) for s in ("UP", "DOWN", "BULL", "BEAR")):
        return False
    return symbol.endswith("USDT")


def _candidate_symbols(pool_size: int) -> list[str]:
    exchange_info = _get("/api/v3/exchangeInfo")
    tickers = _get("/api/v3/ticker/24hr")

    valid = {
        item["symbol"]
        for item in exchange_info.get("symbols", [])
        if _valid_usdt_symbol(item)
    }

    rows = []
    for item in tickers:
        symbol = item.get("symbol")
        if symbol not in valid:
            continue
        rows.append(
            (
                symbol,
                float(item.get("quoteVolume") or 0.0),
            )
        )

    rows.sort(key=lambda x: x[1], reverse=True)
    return [symbol for symbol, _ in rows[: max(10, int(pool_size))]]


def _fetch_hourly_history(symbol: str, start_ms: int, end_ms: int) -> pd.DataFrame:
    rows: list[list] = []
    cursor = int(start_ms)

    while cursor < end_ms:
        batch = _get(
            "/api/v3/klines",
            {
                "symbol": symbol,
                "interval": "1h",
                "startTime": cursor,
                "endTime": int(end_ms),
                "limit": 1000,
            },
        )
        if not isinstance(batch, list) or not batch:
            break

        rows.extend(batch)
        last_open = int(batch[-1][0])
        next_cursor = last_open + 3_600_000
        if next_cursor <= cursor:
            break
        cursor = next_cursor

        if len(batch) < 1000:
            break
        time.sleep(0.03)

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


def _atr_pct(df: pd.DataFrame, period: int = 14) -> float:
    if len(df) < period + 2:
        return float("nan")
    prev_close = df["close"].shift(1)
    tr = pd.concat(
        [
            df["high"] - df["low"],
            (df["high"] - prev_close).abs(),
            (df["low"] - prev_close).abs(),
        ],
        axis=1,
    ).max(axis=1)
    atr = tr.rolling(period, min_periods=period).mean()
    values = (atr / df["close"].replace(0.0, float("nan")) * 100.0).dropna()
    return float(values.median()) if not values.empty else float("nan")


def _window_metrics(df: pd.DataFrame) -> dict | None:
    if df.empty or len(df) < 48:
        return None

    returns = df["close"].pct_change().dropna()
    if returns.empty:
        return None

    days = max(
        (df.index[-1] - df.index[0]).total_seconds() / 86_400.0,
        1.0,
    )
    realized_daily_pct = float(returns.std()) * math.sqrt(24) * 100.0
    avg_abs_1h_pct = float(returns.abs().mean()) * 100.0
    avg_daily_turnover = float(df["quote_volume"].sum()) / days
    avg_daily_trades = float(df["trades"].sum()) / days

    return {
        "realized_vol_pct": realized_daily_pct,
        "atr_pct": _atr_pct(df),
        "avg_abs_1h_pct": avg_abs_1h_pct,
        "avg_daily_turnover": avg_daily_turnover,
        "avg_daily_trades": avg_daily_trades,
    }


def _rank_score(frame: pd.DataFrame) -> pd.DataFrame:
    out = frame.copy()
    out["liq_rank"] = out["avg_daily_turnover"].rank(pct=True) * 100.0
    out["vol_rank"] = out["realized_vol_pct"].rank(pct=True) * 100.0
    out["atr_rank"] = out["atr_pct"].rank(pct=True) * 100.0
    out["activity_rank"] = out["avg_daily_trades"].rank(pct=True) * 100.0

    out["score"] = (
        0.30 * out["liq_rank"]
        + 0.35 * out["vol_rank"]
        + 0.25 * out["atr_rank"]
        + 0.10 * out["activity_rank"]
    )
    return out


def run_rolling_universe_validation(
    *,
    horizon_days: int = 90,
    lookback_days: int = 14,
    forward_days: int = 7,
    pool_size: int = 20,
    select_top_n: int = 5,
    min_daily_turnover_usd: float = 5_000_000.0,
) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    """Validate whether the screener predicts next-period opportunity.

    Each rebalance ranks symbols using only the trailing lookback window and
    evaluates the selected set on the following forward window.

    Research-stage caveat: the candidate pool is restricted to symbols that are
    listed/trading today, so this does not remove survivorship bias completely.
    """
    horizon_days = max(30, min(int(horizon_days), 365))
    lookback_days = max(7, min(int(lookback_days), 60))
    forward_days = max(1, min(int(forward_days), 14))
    pool_size = max(10, min(int(pool_size), 50))
    select_top_n = max(1, min(int(select_top_n), 10))

    now = pd.Timestamp.now(tz="UTC").floor("h")
    analysis_start = now - pd.Timedelta(days=horizon_days)
    history_start = analysis_start - pd.Timedelta(days=lookback_days + 2)

    symbols = _candidate_symbols(pool_size)
    history: dict[str, pd.DataFrame] = {}

    start_ms = int(history_start.timestamp() * 1000)
    end_ms = int(now.timestamp() * 1000)

    for symbol in symbols:
        try:
            history[symbol] = _fetch_hourly_history(symbol, start_ms, end_ms)
        except Exception:
            history[symbol] = pd.DataFrame()
        time.sleep(0.02)

    period_rows = []
    detail_rows = []

    selection_time = analysis_start
    period_number = 0

    while selection_time + pd.Timedelta(days=forward_days) <= now:
        period_number += 1
        lookback_start = selection_time - pd.Timedelta(days=lookback_days)
        forward_end = selection_time + pd.Timedelta(days=forward_days)

        in_sample_rows = []
        for symbol, df in history.items():
            if df.empty:
                continue
            window = df[(df.index >= lookback_start) & (df.index < selection_time)]
            metrics = _window_metrics(window)
            if metrics is None:
                continue
            if metrics["avg_daily_turnover"] < float(min_daily_turnover_usd):
                continue

            in_sample_rows.append({"Symbol": symbol, **metrics})

        if len(in_sample_rows) < max(3, select_top_n):
            selection_time += pd.Timedelta(days=forward_days)
            continue

        ranked = _rank_score(pd.DataFrame(in_sample_rows)).sort_values(
            "score",
            ascending=False,
        )
        selected_symbols = ranked.head(select_top_n)["Symbol"].tolist()

        forward_rows = []
        for symbol in ranked["Symbol"].tolist():
            df = history.get(symbol, pd.DataFrame())
            forward = df[
                (df.index >= selection_time)
                & (df.index < forward_end)
            ]
            metrics = _window_metrics(forward)
            if metrics is None:
                continue

            selected = symbol in selected_symbols
            forward_rows.append(
                {
                    "Symbol": symbol,
                    "Selected": selected,
                    **metrics,
                }
            )

            if selected:
                source = ranked[ranked["Symbol"] == symbol].iloc[0]
                detail_rows.append(
                    {
                        "Period": period_number,
                        "Selection time": selection_time,
                        "Symbol": symbol,
                        "Selection score": float(source["score"]),
                        "Lookback vol %": float(source["realized_vol_pct"]),
                        "Lookback ATR %": float(source["atr_pct"]),
                        "Lookback daily turnover $": float(source["avg_daily_turnover"]),
                        "Forward vol %": float(metrics["realized_vol_pct"]),
                        "Forward ATR %": float(metrics["atr_pct"]),
                        "Forward abs 1h move %": float(metrics["avg_abs_1h_pct"]),
                        "Forward daily turnover $": float(metrics["avg_daily_turnover"]),
                    }
                )

        forward_df = pd.DataFrame(forward_rows)
        if forward_df.empty or not forward_df["Selected"].any():
            selection_time += pd.Timedelta(days=forward_days)
            continue

        selected_df = forward_df[forward_df["Selected"]]
        universe_df = forward_df

        sel_vol = float(selected_df["realized_vol_pct"].mean())
        uni_vol = float(universe_df["realized_vol_pct"].mean())
        sel_move = float(selected_df["avg_abs_1h_pct"].mean())
        uni_move = float(universe_df["avg_abs_1h_pct"].mean())
        sel_turnover = float(selected_df["avg_daily_turnover"].mean())
        uni_turnover = float(universe_df["avg_daily_turnover"].mean())

        vol_uplift = (sel_vol / uni_vol - 1.0) * 100.0 if uni_vol > 0 else 0.0
        move_uplift = (sel_move / uni_move - 1.0) * 100.0 if uni_move > 0 else 0.0
        turnover_uplift = (
            (sel_turnover / uni_turnover - 1.0) * 100.0
            if uni_turnover > 0
            else 0.0
        )

        period_rows.append(
            {
                "Period": period_number,
                "Selection time": selection_time,
                "Selected": ", ".join(selected_symbols),
                "Eligible symbols": int(len(ranked)),
                "Selected forward vol %": sel_vol,
                "Universe forward vol %": uni_vol,
                "Vol uplift %": vol_uplift,
                "Selected abs 1h move %": sel_move,
                "Universe abs 1h move %": uni_move,
                "Move uplift %": move_uplift,
                "Selected daily turnover $": sel_turnover,
                "Universe daily turnover $": uni_turnover,
                "Turnover uplift %": turnover_uplift,
            }
        )

        selection_time += pd.Timedelta(days=forward_days)

    periods = pd.DataFrame(period_rows)
    details = pd.DataFrame(detail_rows)

    if periods.empty:
        summary = {
            "periods": 0,
            "positive_vol_uplift_periods": 0,
            "avg_vol_uplift_pct": 0.0,
            "median_vol_uplift_pct": 0.0,
            "avg_move_uplift_pct": 0.0,
            "avg_turnover_uplift_pct": 0.0,
        }
    else:
        summary = {
            "periods": int(len(periods)),
            "positive_vol_uplift_periods": int((periods["Vol uplift %"] > 0).sum()),
            "avg_vol_uplift_pct": float(periods["Vol uplift %"].mean()),
            "median_vol_uplift_pct": float(periods["Vol uplift %"].median()),
            "avg_move_uplift_pct": float(periods["Move uplift %"].mean()),
            "avg_turnover_uplift_pct": float(periods["Turnover uplift %"].mean()),
        }

    return periods, details, summary
