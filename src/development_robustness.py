from __future__ import annotations

import numpy as np
import pandas as pd


FEATURES = [
    "1m volume acceleration",
    "1m trade acceleration",
    "1m body / range",
    "1m close location",
    "1m impulse 3m bps",
    "1m realized vol 5m bps",
    "1m taker buy ratio",
    "1m taker buy ratio delta",
    "5m volume acceleration",
    "5m close location",
    "Breadth acceleration 15m",
    "Relative extension / ATR",
    "Impulse age 5m bars",
    "5m taker buy ratio",
    "5m taker buy ratio delta",
]

HORIZONS = [3, 5, 10, 15]
PRIMARY_OUTCOME = "Gross 10m bps"


def _baseline(details: pd.DataFrame) -> pd.DataFrame:
    base = details[
        details["Tier"].astype(str) == "Baseline LONG/BULL"
    ].copy()
    return (
        base.drop_duplicates(
            subset=["Period", "Symbol", "Event time"],
            keep="first",
        )
        .reset_index(drop=True)
    )


def _safe_spearman(x: pd.Series, y: pd.Series, min_n: int = 12) -> float:
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


def _quartile_spread(frame: pd.DataFrame, feature: str, outcome: str) -> float:
    work = frame[[feature, outcome]].copy()
    work[feature] = pd.to_numeric(work[feature], errors="coerce")
    work[outcome] = pd.to_numeric(work[outcome], errors="coerce")
    work = work.dropna()

    if len(work) < 20 or work[feature].nunique() < 4:
        return float("nan")

    q1 = float(work[feature].quantile(0.25))
    q3 = float(work[feature].quantile(0.75))
    low = work[work[feature] <= q1][outcome]
    high = work[work[feature] >= q3][outcome]

    if low.empty or high.empty:
        return float("nan")

    return float(high.mean() - low.mean())


def _same_sign_count(values: list[float], target_sign: float) -> tuple[int, int]:
    valid = [float(v) for v in values if np.isfinite(v) and np.sign(v) != 0]
    if not valid or target_sign == 0:
        return 0, len(valid)
    same = sum(1 for value in valid if np.sign(value) == target_sign)
    return same, len(valid)


def _time_block_correlations(base: pd.DataFrame, feature: str) -> list[float]:
    periods = sorted(base["Period"].dropna().unique().tolist())
    if not periods:
        return []

    blocks = [list(chunk) for chunk in np.array_split(periods, min(4, len(periods))) if len(chunk)]
    correlations = []
    for block in blocks:
        part = base[base["Period"].isin(block)]
        correlations.append(
            _safe_spearman(part[feature], part[PRIMARY_OUTCOME], min_n=10)
        )
    return correlations


def _horizon_correlations(base: pd.DataFrame, feature: str) -> list[float]:
    out = []
    for horizon in HORIZONS:
        outcome = f"Gross {horizon}m bps"
        if outcome not in base.columns:
            continue
        out.append(_safe_spearman(base[feature], base[outcome], min_n=12))
    return out


def _jackknife_correlations(base: pd.DataFrame, feature: str) -> list[float]:
    symbols = (
        base["Symbol"]
        .astype(str)
        .value_counts()
        .loc[lambda x: x >= 3]
        .index
        .tolist()
    )
    values = []
    for symbol in symbols:
        part = base[base["Symbol"].astype(str) != symbol]
        values.append(
            _safe_spearman(part[feature], part[PRIMARY_OUTCOME], min_n=25)
        )
    return values


def _favorable_tail_top_symbol_share(
    base: pd.DataFrame,
    feature: str,
    effect_sign: float,
) -> float:
    work = base[[feature, "Symbol"]].copy()
    work[feature] = pd.to_numeric(work[feature], errors="coerce")
    work = work.dropna()
    if len(work) < 20 or effect_sign == 0:
        return 100.0

    q1 = float(work[feature].quantile(0.25))
    q3 = float(work[feature].quantile(0.75))
    if effect_sign > 0:
        tail = work[work[feature] >= q3]
    else:
        tail = work[work[feature] <= q1]

    if tail.empty:
        return 100.0

    counts = tail["Symbol"].astype(str).value_counts()
    return float(counts.iloc[0] / counts.sum() * 100.0)


def analyze_development_robustness(
    development_details: pd.DataFrame,
) -> tuple[pd.DataFrame, str]:
    """Development-only robustness screen.

    This does not optimize a trading rule. A feature must show the same broad
    relationship across chronological blocks, multiple horizons, and
    leave-one-symbol-out checks before it is allowed to generate a new
    hypothesis for a future untouched window.
    """
    if development_details is None or development_details.empty:
        raise ValueError("Development event details are missing.")

    base = _baseline(development_details)
    if base.empty:
        raise RuntimeError("No Baseline LONG/BULL development events are available.")
    if PRIMARY_OUTCOME not in base.columns:
        raise RuntimeError("10m forward outcome is missing from development events.")

    available = [feature for feature in FEATURES if feature in base.columns]
    if not available:
        raise RuntimeError("No v2.8 robustness features are available.")

    rows = []
    for feature in available:
        complete = base[[feature, PRIMARY_OUTCOME, "Period", "Symbol"]].copy()
        complete[feature] = pd.to_numeric(complete[feature], errors="coerce")
        complete[PRIMARY_OUTCOME] = pd.to_numeric(
            complete[PRIMARY_OUTCOME], errors="coerce"
        )
        complete = complete.dropna()

        if len(complete) < 30:
            continue

        overall_rho = _safe_spearman(
            complete[feature],
            complete[PRIMARY_OUTCOME],
            min_n=20,
        )
        spread = _quartile_spread(complete, feature, PRIMARY_OUTCOME)
        effect_sign = (
            float(np.sign(overall_rho))
            if np.isfinite(overall_rho) and overall_rho != 0
            else 0.0
        )

        time_corrs = _time_block_correlations(base, feature)
        horizon_corrs = _horizon_correlations(base, feature)
        jackknife_corrs = _jackknife_correlations(base, feature)

        time_same, time_total = _same_sign_count(time_corrs, effect_sign)
        horizon_same, horizon_total = _same_sign_count(horizon_corrs, effect_sign)
        jack_same, jack_total = _same_sign_count(jackknife_corrs, effect_sign)

        jackknife_pct = (
            float(jack_same / jack_total * 100.0)
            if jack_total
            else 0.0
        )

        block_valid = [x for x in time_corrs if np.isfinite(x)]
        median_block_rho = (
            float(np.median(block_valid))
            if block_valid
            else float("nan")
        )

        top_share = _favorable_tail_top_symbol_share(
            base,
            feature,
            effect_sign,
        )

        spread_agrees = (
            np.isfinite(spread)
            and effect_sign != 0
            and np.sign(spread) == effect_sign
        )

        robust = (
            np.isfinite(overall_rho)
            and abs(float(overall_rho)) >= 0.12
            and spread_agrees
            and abs(float(spread)) >= 12.0
            and time_total >= 3
            and time_same >= 3
            and np.isfinite(median_block_rho)
            and np.sign(median_block_rho) == effect_sign
            and horizon_total >= 3
            and horizon_same >= 3
            and jack_total >= 5
            and jackknife_pct >= 80.0
            and top_share <= 35.0
        )

        rows.append(
            {
                "Feature": feature,
                "Events": int(len(complete)),
                "Overall Spearman": overall_rho,
                "10m Q4-Q1 spread bps": spread,
                "Time blocks same sign": f"{time_same}/{time_total}",
                "Median block Spearman": median_block_rho,
                "Horizons same sign": f"{horizon_same}/{horizon_total}",
                "Jackknife same sign %": jackknife_pct,
                "Favorable-tail top symbol %": top_share,
                "Robust candidate": "YES" if robust else "NO",
            }
        )

    summary = pd.DataFrame(rows)
    if summary.empty:
        raise RuntimeError("v2.8 could not evaluate any feature robustly.")

    summary["_rank"] = (
        summary["Robust candidate"].eq("YES").astype(int) * 10
        + summary["Overall Spearman"].abs().fillna(0.0)
    )
    summary = (
        summary.sort_values("_rank", ascending=False)
        .drop(columns=["_rank"])
        .reset_index(drop=True)
    )

    robust = summary[summary["Robust candidate"] == "YES"]

    if robust.empty:
        verdict = (
            f"v2.8 checked {len(summary)} development features across chronological blocks, "
            "3/5/10/15m horizons and leave-one-symbol-out tests. None passed the full robustness "
            "screen. The earlier v2.6 relationships were not stable enough to justify opening a "
            "fourth untouched window. Keep the screener, reject more threshold tuning, and add a "
            "different explanatory feature family before another holdout."
        )
    else:
        names = robust["Feature"].head(3).tolist()
        details = []
        for _, row in robust.head(3).iterrows():
            details.append(
                f"{row['Feature']} (rho {float(row['Overall Spearman']):+.2f}, "
                f"10m Q4-Q1 {float(row['10m Q4-Q1 spread bps']):+.1f} bps, "
                f"time {row['Time blocks same sign']}, "
                f"jackknife {float(row['Jackknife same sign %']):.0f}%)"
            )
        verdict = (
            "v2.8 found development features that remained directionally consistent after "
            "time-split and symbol-jackknife checks: "
            + "; ".join(details)
            + ". This still does not validate a strategy. Form one simple economic hypothesis "
            "from these candidates, freeze it, then and only then use a fourth untouched "
            "120-day window ending 510 days before the frozen research anchor."
        )

    return summary, verdict
