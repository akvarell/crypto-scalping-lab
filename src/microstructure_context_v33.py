from __future__ import annotations

import numpy as np
import pandas as pd

from src.dynamic_scalping import _fetch_5m
from src.event_dataset_v3 import _forward_return_bps, _prepare_5m_v3
from src.scalping_edge_map import _fetch_1m


HORIZONS = [3, 5, 10, 15, 30]
SAMPLE_MINUTES = 60

FEATURE_SPECS = [
    ("Order flow", "Taker imbalance"),
    ("Order flow", "Taker imbalance change"),
    ("Order flow", "Volume × taker imbalance"),
    ("Order flow", "Trade-size × taker imbalance"),
    ("Breadth", "Breadth participation"),
    ("Breadth", "Simultaneous highs/lows"),
    ("Breadth", "Context volume participation"),
    ("Dislocation", "Relative-z continuation"),
    ("Dislocation", "Relative-z reversion"),
    ("Microstructure reversal", "5m flow reversal"),
    ("Microstructure reversal", "1m flow-turn reversal"),
    ("Microstructure reversal", "Close-location deterioration"),
]


def _safe_ratio(value: float, baseline: float) -> float:
    if pd.isna(value) or pd.isna(baseline) or float(baseline) == 0.0:
        return float("nan")
    return float(value) / float(baseline)


def _prepare_extended(df: pd.DataFrame) -> pd.DataFrame:
    out = _prepare_5m_v3(df).copy()

    if "quote_volume" in out.columns:
        prior_quote = (
            out["quote_volume"]
            .shift(1)
            .rolling(20, min_periods=20)
            .mean()
        )
        out["quote_volume_ratio"] = (
            out["quote_volume"] / prior_quote.replace(0.0, float("nan"))
        )
    else:
        out["quote_volume_ratio"] = float("nan")

    if "trades" in out.columns and "quote_volume" in out.columns:
        avg_trade = (
            out["quote_volume"]
            / out["trades"].replace(0.0, float("nan"))
        )
        prior_avg_trade = (
            avg_trade.shift(1).rolling(20, min_periods=20).mean()
        )
        out["avg_trade_size_ratio"] = (
            avg_trade / prior_avg_trade.replace(0.0, float("nan"))
        )
    else:
        out["avg_trade_size_ratio"] = float("nan")

    candle_range = out["high"] - out["low"]
    out["close_location"] = (
        (out["close"] - out["low"])
        / candle_range.replace(0.0, float("nan"))
    )
    out["close_location_change"] = out["close_location"].diff()

    out["volume_ratio_change"] = out["volume_ratio"].diff()
    out["trades_ratio_change"] = out["trades_ratio"].diff()

    return out


def _context_frame(
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
    highs = pd.concat(
        {
            s: (
                prepared[s]["close"] > prepared[s]["prior_high_20"]
            ).astype(float)
            for s in others
        },
        axis=1,
    )
    lows = pd.concat(
        {
            s: (
                prepared[s]["close"] < prepared[s]["prior_low_20"]
            ).astype(float)
            for s in others
        },
        axis=1,
    )

    ctx = pd.DataFrame(index=target_index)
    ctx["basket_ret_15m"] = ret15.median(axis=1).reindex(target_index)
    ctx["basket_ret_60m"] = ret60.median(axis=1).reindex(target_index)
    ctx["dispersion_15m"] = ret15.std(axis=1).reindex(target_index)
    ctx["dispersion_change_15m"] = (
        ctx["dispersion_15m"] - ctx["dispersion_15m"].shift(3)
    )

    ctx["breadth"] = breadth.mean(axis=1).reindex(target_index)
    ctx["breadth_change_5m"] = ctx["breadth"].diff()
    ctx["breadth_change_15m"] = ctx["breadth"] - ctx["breadth"].shift(3)
    ctx["breadth_acceleration"] = (
        ctx["breadth_change_5m"] - ctx["breadth_change_5m"].shift(1)
    )

    ctx["fraction_new_highs"] = highs.mean(axis=1).reindex(target_index)
    ctx["fraction_new_lows"] = lows.mean(axis=1).reindex(target_index)
    ctx["net_extremes"] = (
        ctx["fraction_new_highs"] - ctx["fraction_new_lows"]
    )

    ctx["basket_volume_ratio"] = volume.median(axis=1).reindex(target_index)
    ctx["basket_volume_change"] = (
        ctx["basket_volume_ratio"] - ctx["basket_volume_ratio"].shift(1)
    )
    return ctx


def _one_minute_flow_turn(
    minute_data: pd.DataFrame,
    candle_start: pd.Timestamp,
    known_time: pd.Timestamp,
) -> float:
    if minute_data.empty or "taker_base" not in minute_data.columns:
        return float("nan")

    bars = minute_data[
        (minute_data.index >= candle_start)
        & (minute_data.index < known_time)
    ].copy()
    if len(bars) < 5:
        return float("nan")

    denom = bars["volume"].replace(0.0, float("nan"))
    ratios = (bars["taker_base"] / denom).dropna()
    if len(ratios) < 5:
        return float("nan")

    prior = float(ratios.iloc[:-1].mean())
    last = float(ratios.iloc[-1])
    return last - prior


def _add_feature_row(
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

    record = {
        "Period": int(period),
        "Symbol": symbol,
        "Category": category,
        "Feature": feature,
        "Side": "LONG" if side_sign == 1 else "SHORT",
        "Strength": float(strength),
        "Raw value": float(raw_value) if np.isfinite(raw_value) else float("nan"),
    }
    for horizon, raw_bps in future_returns.items():
        record[f"Gross {horizon}m bps"] = (
            float(side_sign) * float(raw_bps)
            if np.isfinite(raw_bps)
            else float("nan")
        )
    rows.append(record)


def build_microstructure_context_period_v33(
    period_frame: pd.DataFrame,
    *,
    forward_days: int = 7,
    warmup_hours: int = 48,
    sample_minutes: int = SAMPLE_MINUTES,
) -> pd.DataFrame:
    """Build broad, threshold-free v3.3 mechanism observations.

    Sampling is deterministic (hourly by default), not conditioned on the
    feature being studied. This avoids selecting only extreme feature values.
    Every feature is known at the close of a completed 5m candle, with entry
    modeled at the first 1m open at/after known_time.
    """
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
    trade_symbols = (
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
        prepared[symbol] = _prepare_extended(raw)

    if len(prepared) < 8:
        return pd.DataFrame()

    rows: list[dict] = []

    for symbol in trade_symbols:
        coin = prepared.get(symbol)
        if coin is None:
            continue

        ctx = _context_frame(symbol, prepared)
        if ctx.empty:
            continue

        aligned = coin.join(ctx, how="left")
        aligned["relative_ret_15m"] = (
            aligned["ret_15m"] - aligned["basket_ret_15m"]
        )
        aligned["relative_z"] = (
            aligned["relative_ret_15m"]
            / aligned["dispersion_15m"].replace(0.0, float("nan"))
        )
        aligned["relative_atr"] = (
            aligned["relative_ret_15m"]
            / aligned["atr_pct"].replace(0.0, float("nan"))
        )

        sample_index = aligned.index[
            (aligned.index >= selection_time)
            & (aligned.index < period_end)
            & (aligned.index.minute % int(sample_minutes) == 0)
        ]
        if len(sample_index) == 0:
            continue

        # Full-period 1m pull once per trade symbol: much cheaper than one
        # request per observation and bounded by the period checkpoint.
        try:
            minute = _fetch_1m(
                symbol,
                selection_time - pd.Timedelta(minutes=10),
                period_end + pd.Timedelta(minutes=40),
            )
        except Exception:
            minute = pd.DataFrame()

        if minute.empty:
            continue

        for ts in sample_index:
            row = aligned.loc[ts]
            known_time = pd.Timestamp(ts) + pd.Timedelta(minutes=5)

            entry_i = int(
                minute.index.searchsorted(known_time, side="left")
            )
            if entry_i >= len(minute):
                continue

            raw_future: dict[int, float] = {}
            for horizon in HORIZONS:
                raw_future[horizon] = _forward_return_bps(
                    minute,
                    entry_i,
                    1,
                    horizon,
                )

            ret15 = float(row.get("ret_15m", float("nan")))
            basket15 = float(row.get("basket_ret_15m", float("nan")))
            relative = float(row.get("relative_ret_15m", float("nan")))
            relative_z = float(row.get("relative_z", float("nan")))

            taker_ratio = float(
                row.get("taker_buy_ratio", float("nan"))
            )
            taker_delta = float(
                row.get("taker_buy_ratio_delta", float("nan"))
            )
            imbalance = (
                2.0 * taker_ratio - 1.0
                if np.isfinite(taker_ratio)
                else float("nan")
            )
            imbalance_delta = (
                2.0 * taker_delta
                if np.isfinite(taker_delta)
                else float("nan")
            )

            vol_ratio = float(row.get("volume_ratio", float("nan")))
            avg_trade_ratio = float(
                row.get("avg_trade_size_ratio", float("nan"))
            )
            flow_turn_1m = _one_minute_flow_turn(
                minute,
                pd.Timestamp(ts),
                known_time,
            )

            # 1) Order-flow imbalance: direction comes from the sign of flow,
            # while strength stays continuous.
            if np.isfinite(imbalance) and imbalance != 0:
                flow_side = 1 if imbalance > 0 else -1
                _add_feature_row(
                    rows,
                    period=period,
                    symbol=symbol,
                    category="Order flow",
                    feature="Taker imbalance",
                    side_sign=flow_side,
                    strength=abs(imbalance),
                    raw_value=imbalance,
                    future_returns=raw_future,
                )
                if np.isfinite(vol_ratio) and vol_ratio > 0:
                    _add_feature_row(
                        rows,
                        period=period,
                        symbol=symbol,
                        category="Order flow",
                        feature="Volume × taker imbalance",
                        side_sign=flow_side,
                        strength=abs(imbalance) * float(np.log1p(vol_ratio)),
                        raw_value=imbalance * float(np.log1p(vol_ratio)),
                        future_returns=raw_future,
                    )
                if np.isfinite(avg_trade_ratio) and avg_trade_ratio > 0:
                    _add_feature_row(
                        rows,
                        period=period,
                        symbol=symbol,
                        category="Order flow",
                        feature="Trade-size × taker imbalance",
                        side_sign=flow_side,
                        strength=abs(imbalance) * float(np.log1p(avg_trade_ratio)),
                        raw_value=imbalance * float(np.log1p(avg_trade_ratio)),
                        future_returns=raw_future,
                    )

            if np.isfinite(imbalance_delta) and imbalance_delta != 0:
                delta_side = 1 if imbalance_delta > 0 else -1
                _add_feature_row(
                    rows,
                    period=period,
                    symbol=symbol,
                    category="Order flow",
                    feature="Taker imbalance change",
                    side_sign=delta_side,
                    strength=abs(imbalance_delta),
                    raw_value=imbalance_delta,
                    future_returns=raw_future,
                )

            # 2) Breadth/participation. We only require directional agreement,
            # not a magnitude threshold.
            breadth15 = float(
                row.get("breadth_change_15m", float("nan"))
            )
            if np.isfinite(basket15) and basket15 != 0 and np.isfinite(breadth15):
                basket_side = 1 if basket15 > 0 else -1
                participation = basket_side * breadth15
                if participation > 0:
                    _add_feature_row(
                        rows,
                        period=period,
                        symbol=symbol,
                        category="Breadth",
                        feature="Breadth participation",
                        side_sign=basket_side,
                        strength=participation,
                        raw_value=breadth15,
                        future_returns=raw_future,
                    )

                context_volume = float(
                    row.get("basket_volume_ratio", float("nan"))
                )
                if np.isfinite(context_volume) and context_volume > 0:
                    _add_feature_row(
                        rows,
                        period=period,
                        symbol=symbol,
                        category="Breadth",
                        feature="Context volume participation",
                        side_sign=basket_side,
                        strength=context_volume,
                        raw_value=context_volume,
                        future_returns=raw_future,
                    )

            net_extremes = float(row.get("net_extremes", float("nan")))
            if np.isfinite(net_extremes) and net_extremes != 0:
                extreme_side = 1 if net_extremes > 0 else -1
                _add_feature_row(
                    rows,
                    period=period,
                    symbol=symbol,
                    category="Breadth",
                    feature="Simultaneous highs/lows",
                    side_sign=extreme_side,
                    strength=abs(net_extremes),
                    raw_value=net_extremes,
                    future_returns=raw_future,
                )

            # 3) Same dislocation tested as continuation and mean reversion.
            if np.isfinite(relative_z) and relative_z != 0:
                rel_side = 1 if relative_z > 0 else -1
                _add_feature_row(
                    rows,
                    period=period,
                    symbol=symbol,
                    category="Dislocation",
                    feature="Relative-z continuation",
                    side_sign=rel_side,
                    strength=abs(relative_z),
                    raw_value=relative_z,
                    future_returns=raw_future,
                )
                _add_feature_row(
                    rows,
                    period=period,
                    symbol=symbol,
                    category="Dislocation",
                    feature="Relative-z reversion",
                    side_sign=-rel_side,
                    strength=abs(relative_z),
                    raw_value=relative_z,
                    future_returns=raw_future,
                )

            # 4) Reversal evidence must oppose the completed price move.
            move = relative if np.isfinite(relative) and relative != 0 else ret15
            if np.isfinite(move) and move != 0:
                move_sign = 1 if move > 0 else -1
                reversal_side = -move_sign

                if np.isfinite(imbalance_delta):
                    reversal_strength = -move_sign * imbalance_delta
                    if reversal_strength > 0:
                        _add_feature_row(
                            rows,
                            period=period,
                            symbol=symbol,
                            category="Microstructure reversal",
                            feature="5m flow reversal",
                            side_sign=reversal_side,
                            strength=reversal_strength,
                            raw_value=imbalance_delta,
                            future_returns=raw_future,
                        )

                if np.isfinite(flow_turn_1m):
                    turn_strength = -move_sign * flow_turn_1m
                    if turn_strength > 0:
                        _add_feature_row(
                            rows,
                            period=period,
                            symbol=symbol,
                            category="Microstructure reversal",
                            feature="1m flow-turn reversal",
                            side_sign=reversal_side,
                            strength=turn_strength,
                            raw_value=flow_turn_1m,
                            future_returns=raw_future,
                        )

                close_change = float(
                    row.get("close_location_change", float("nan"))
                )
                if np.isfinite(close_change):
                    deterioration = -move_sign * close_change
                    if deterioration > 0:
                        _add_feature_row(
                            rows,
                            period=period,
                            symbol=symbol,
                            category="Microstructure reversal",
                            feature="Close-location deterioration",
                            side_sign=reversal_side,
                            strength=deterioration,
                            raw_value=close_change,
                            future_returns=raw_future,
                        )

    return pd.DataFrame(rows)


def _spearman(x: pd.Series, y: pd.Series, min_n: int = 30) -> float:
    pair = pd.concat(
        [
            pd.to_numeric(x, errors="coerce"),
            pd.to_numeric(y, errors="coerce"),
        ],
        axis=1,
    ).dropna()
    if len(pair) < min_n:
        return float("nan")
    if pair.iloc[:, 0].nunique() < 4 or pair.iloc[:, 1].nunique() < 4:
        return float("nan")
    return float(
        pair.iloc[:, 0].rank(method="average").corr(
            pair.iloc[:, 1].rank(method="average")
        )
    )


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


def _quartile_stats(
    frame: pd.DataFrame,
    outcome: str,
) -> dict:
    work = frame[["Strength", outcome, "Period", "Symbol"]].copy()
    work["Strength"] = pd.to_numeric(work["Strength"], errors="coerce")
    work[outcome] = pd.to_numeric(work[outcome], errors="coerce")
    work = work.dropna()

    if len(work) < 80 or work["Strength"].nunique() < 4:
        return {}

    q1 = float(work["Strength"].quantile(0.25))
    q3 = float(work["Strength"].quantile(0.75))
    low = work[work["Strength"] <= q1]
    high = work[work["Strength"] >= q3]
    if low.empty or high.empty:
        return {}

    high_period = high.groupby("Period")[outcome].mean()
    top_share = float(
        high["Symbol"].value_counts().iloc[0] / len(high) * 100.0
    )

    return {
        "spread": float(high[outcome].mean() - low[outcome].mean()),
        "q4_avg": float(high[outcome].mean()),
        "q4_median": float(high[outcome].median()),
        "q4_trimmed": _trimmed_mean(high[outcome]),
        "q4_positive_periods": (
            f"{int((high_period > 0).sum())}/{len(high_period)}"
            if len(high_period)
            else "0/0"
        ),
        "q4_top_symbol_share": top_share,
        "q4_n": int(len(high)),
    }


def _cluster_spread_ci(
    frame: pd.DataFrame,
    outcome: str,
    *,
    repeats: int = 400,
    seed: int = 45,
) -> tuple[float, float]:
    periods = sorted(frame["Period"].dropna().unique().tolist())
    if len(periods) < 8:
        return float("nan"), float("nan")

    rng = np.random.default_rng(seed)
    values: list[float] = []

    for _ in range(int(repeats)):
        sampled = rng.choice(periods, size=len(periods), replace=True)
        parts = []
        for block_id, period in enumerate(sampled):
            part = frame[frame["Period"] == period][
                ["Strength", outcome]
            ].copy()
            if not part.empty:
                part["_block"] = block_id
                parts.append(part)

        if not parts:
            continue

        boot = pd.concat(parts, ignore_index=True)
        stats = _quartile_stats(
            boot.assign(Period=boot["_block"], Symbol="bootstrap"),
            outcome,
        )
        if stats and np.isfinite(stats["spread"]):
            values.append(float(stats["spread"]))

    if len(values) < 100:
        return float("nan"), float("nan")

    return (
        float(np.quantile(values, 0.005)),
        float(np.quantile(values, 0.995)),
    )


def _fraction(value: object) -> float:
    try:
        left, right = str(value).split("/", 1)
        denom = int(right)
        return int(left) / denom if denom else 0.0
    except Exception:
        return 0.0


def analyze_microstructure_context_v33(
    observations: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, str]:
    """Discovery-only continuous relationship screen.

    No trading threshold is optimized. Strength is evaluated continuously,
    then the upper/lower quartiles are used only to describe effect size.
    """
    if observations is None or observations.empty:
        raise ValueError("v3.3 observations are empty.")

    rows: list[dict] = []

    for (category, feature, side), frame in observations.groupby(
        ["Category", "Feature", "Side"],
        sort=False,
    ):
        if len(frame) < 200:
            continue

        for horizon in HORIZONS:
            outcome = f"Gross {horizon}m bps"
            work = frame[
                ["Period", "Symbol", "Strength", outcome]
            ].copy()
            work = work.dropna()
            if len(work) < 200:
                continue

            rho = _spearman(work["Strength"], work[outcome], min_n=100)
            q = _quartile_stats(work, outcome)
            if not q:
                continue

            period_rhos = []
            for period in sorted(work["Period"].unique().tolist()):
                part = work[work["Period"] == period]
                prho = _spearman(
                    part["Strength"],
                    part[outcome],
                    min_n=15,
                )
                if np.isfinite(prho):
                    period_rhos.append(float(prho))

            same_sign_periods = (
                sum(1 for value in period_rhos if value > 0)
                if np.isfinite(rho) and rho > 0
                else sum(1 for value in period_rhos if value < 0)
            )

            loo = []
            periods = sorted(work["Period"].unique().tolist())
            for period in periods:
                part = work[work["Period"] != period]
                lrho = _spearman(
                    part["Strength"],
                    part[outcome],
                    min_n=100,
                )
                if np.isfinite(lrho):
                    loo.append(float(lrho))

            same_loo = (
                sum(1 for value in loo if value > 0)
                if np.isfinite(rho) and rho > 0
                else sum(1 for value in loo if value < 0)
            )

            ci_low, ci_high = _cluster_spread_ci(work, outcome)

            candidate = (
                np.isfinite(rho)
                and rho >= 0.08
                and q["spread"] >= 12.0
                and q["q4_median"] >= 12.0
                and q["q4_trimmed"] >= 12.0
                and _fraction(q["q4_positive_periods"]) > 0.5
                and q["q4_top_symbol_share"] <= 35.0
                and len(period_rhos) >= 8
                and same_sign_periods / len(period_rhos) >= 0.65
                and len(loo) >= 8
                and same_loo / len(loo) >= 0.80
                and np.isfinite(ci_low)
                and ci_low > 0
            )

            rows.append(
                {
                    "Category": category,
                    "Feature": feature,
                    "Side": side,
                    "Horizon min": int(horizon),
                    "Events": int(len(work)),
                    "Periods": int(work["Period"].nunique()),
                    "Symbols": int(work["Symbol"].nunique()),
                    "Spearman strength→return": rho,
                    "Q4-Q1 gross spread bps": q["spread"],
                    "Q4 gross avg bps": q["q4_avg"],
                    "Q4 gross median bps": q["q4_median"],
                    "Q4 trimmed gross bps": q["q4_trimmed"],
                    "Q4 positive periods": q["q4_positive_periods"],
                    "Q4 top symbol share %": q["q4_top_symbol_share"],
                    "99% cluster spread CI low": ci_low,
                    "99% cluster spread CI high": ci_high,
                    "Same-sign period correlations": (
                        f"{same_sign_periods}/{len(period_rhos)}"
                    ),
                    "Leave-one-period same sign": f"{same_loo}/{len(loo)}",
                    "Discovery candidate": "YES" if candidate else "NO",
                }
            )

    surface = pd.DataFrame(rows)
    if surface.empty:
        raise RuntimeError("v3.3 produced no relationship surface.")

    # Require support on a neighboring horizon to reduce isolated multiple-test hits.
    candidates = surface[surface["Discovery candidate"] == "YES"].copy()
    if not candidates.empty:
        neighbor_map = {
            3: [5],
            5: [3, 10],
            10: [5, 15],
            15: [10, 30],
            30: [15],
        }
        support = []
        for _, row in candidates.iterrows():
            peers = surface[
                (surface["Feature"] == row["Feature"])
                & (surface["Side"] == row["Side"])
                & (surface["Horizon min"].isin(
                    neighbor_map[int(row["Horizon min"])]
                ))
                & (surface["Spearman strength→return"] > 0)
                & (surface["Q4-Q1 gross spread bps"] > 0)
                & (surface["Q4 gross median bps"] > 0)
            ]
            support.append("YES" if not peers.empty else "NO")
        candidates["Neighbor support"] = support

    if candidates.empty:
        best = surface.sort_values(
            ["Q4 trimmed gross bps", "Spearman strength→return"],
            ascending=False,
        ).iloc[0]
        verdict = (
            "No microstructure/context relationship passed the strict discovery screen. "
            f"The strongest descriptive relationship was {best['Feature']} / {best['Side']} / "
            f"{int(best['Horizon min'])}m: rho {float(best['Spearman strength→return']):+.2f}, "
            f"Q4-Q1 {float(best['Q4-Q1 gross spread bps']):+.1f} bps, "
            f"Q4 median {float(best['Q4 gross median bps']):+.1f} bps, "
            f"positive periods {best['Q4 positive periods']}, 99% spread-CI low "
            f"{float(best['99% cluster spread CI low']):+.1f} bps. "
            "This is diagnostic only; do not derive a trading threshold from it."
        )
    else:
        supported = candidates[
            candidates["Neighbor support"].astype(str) == "YES"
        ]
        shown = supported if not supported.empty else candidates
        parts = []
        for _, row in shown.sort_values(
            "Q4 trimmed gross bps",
            ascending=False,
        ).head(3).iterrows():
            parts.append(
                f"{row['Feature']} / {row['Side']} / {int(row['Horizon min'])}m "
                f"(rho {float(row['Spearman strength→return']):+.2f}, "
                f"Q4-Q1 {float(row['Q4-Q1 gross spread bps']):+.1f} bps, "
                f"Q4 median {float(row['Q4 gross median bps']):+.1f}, "
                f"periods {row['Q4 positive periods']}, "
                f"99% CI low {float(row['99% cluster spread CI low']):+.1f}, "
                f"neighbor {row.get('Neighbor support', 'NO')})"
            )
        verdict = (
            "v3.3 found development-only microstructure/context candidates: "
            + "; ".join(parts)
            + ". No threshold has been validated. The next step is v3.4 purged walk-forward, "
            "where any threshold must be derived inside each training fold only."
        )

    return surface, candidates, verdict
