from __future__ import annotations

import numpy as np
import pandas as pd


FEATURES_V3 = [
    "Breadth",
    "Basket 15m return %",
    "Basket 60m return %",
    "Basket volume ratio",
    "5m volume ratio prior20",
    "5m trades ratio prior20",
    "5m range / ATR",
    "5m relative move / ATR",
    "5m ATR %",
    "5m RSI",
    "5m close location",
    "5m taker buy ratio",
    "5m taker buy ratio delta",
    "1m volume ratio prior20",
    "1m trades ratio prior20",
    "1m body / range",
    "1m close location",
    "1m impulse 3m bps",
    "1m realized vol prior5 bps",
    "1m taker buy ratio",
    "1m taker buy ratio delta",
]


def _spearman(x: pd.Series, y: pd.Series, min_n: int = 12) -> float:
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


def audit_event_dataset_v3(
    universe_details: pd.DataFrame,
    events: pd.DataFrame,
) -> tuple[pd.DataFrame, str]:
    if universe_details is None or universe_details.empty:
        raise ValueError("v3 universe is empty.")
    if events is None or events.empty:
        raise ValueError("v3 event dataset is empty.")

    checks = []

    invalid_before_selection = int(
        (
            pd.to_datetime(events["Event time"], utc=True)
            < pd.to_datetime(events["Selection time"], utc=True)
        ).sum()
    )
    checks.append(
        {
            "Check": "Events after selection time",
            "Value": invalid_before_selection,
            "Status": "PASS" if invalid_before_selection == 0 else "FAIL",
        }
    )

    invalid_execution = int(
        (
            pd.to_datetime(events["Entry time"], utc=True)
            <= pd.to_datetime(events["Known time"], utc=True)
        ).sum()
    )
    checks.append(
        {
            "Check": "Entry strictly after information time",
            "Value": invalid_execution,
            "Status": "PASS" if invalid_execution == 0 else "FAIL",
        }
    )

    context_counts = (
        universe_details[universe_details["Context selected"]]
        .groupby("Period")["Symbol"]
        .nunique()
    )
    min_context = int(context_counts.min()) if not context_counts.empty else 0
    checks.append(
        {
            "Check": "Minimum context basket size",
            "Value": min_context,
            "Status": "PASS" if min_context >= 8 else "FAIL",
        }
    )

    event_periods = int(events["Period"].nunique())
    event_symbols = int(events["Symbol"].nunique())
    checks.append(
        {
            "Check": "Event coverage",
            "Value": f"{len(events)} events / {event_periods} periods / {event_symbols} symbols",
            "Status": (
                "PASS"
                if len(events) >= 100 and event_periods >= 8 and event_symbols >= 8
                else "REVIEW"
            ),
        }
    )

    prior_volume_coverage = (
        float(pd.to_numeric(events["5m volume ratio prior20"], errors="coerce").notna().mean() * 100.0)
        if "5m volume ratio prior20" in events
        else 0.0
    )
    checks.append(
        {
            "Check": "Prior-only 5m volume feature coverage %",
            "Value": round(prior_volume_coverage, 1),
            "Status": "PASS" if prior_volume_coverage >= 95.0 else "REVIEW",
        }
    )

    audit = pd.DataFrame(checks)
    failed = audit[audit["Status"] == "FAIL"]

    verdict = (
        "v3 data-timing audit passed its hard checks."
        if failed.empty
        else "v3 data-timing audit found a hard failure; do not evaluate strategy features yet."
    )
    verdict += (
        " Historical symbol selection is still conditional on coins listed today, because "
        "Binance's standard public exchangeInfo endpoint does not reconstruct delisted symbols."
    )

    return audit, verdict


def cost_stress_v3(
    events: pd.DataFrame,
    *,
    gross_col: str = "Gross 10m bps",
    costs_bps: tuple[float, ...] = (12.0, 20.0, 30.0),
) -> pd.DataFrame:
    rows = []

    for side in ["ALL", "LONG", "SHORT"]:
        group = events if side == "ALL" else events[events["Side"] == side]
        if group.empty:
            continue

        gross = pd.to_numeric(group[gross_col], errors="coerce")
        valid = group.loc[gross.notna()].copy()
        valid[gross_col] = gross.dropna().values

        for cost in costs_bps:
            net = valid[gross_col] - float(cost)
            period_avg = (
                valid.assign(_net=net)
                .groupby("Period")["_net"]
                .mean()
            )

            rows.append(
                {
                    "Side": side,
                    "Round-trip cost bps": float(cost),
                    "Events": int(len(valid)),
                    "Periods": int(valid["Period"].nunique()),
                    "Gross avg bps": float(valid[gross_col].mean()),
                    "Gross median bps": float(valid[gross_col].median()),
                    "Trimmed gross avg bps": _trimmed_mean(valid[gross_col]),
                    "Net avg bps": float(net.mean()),
                    "Net median bps": float(net.median()),
                    "Net positive events %": float((net > 0).mean() * 100.0),
                    "Positive periods": (
                        f"{int((period_avg > 0).sum())}/{len(period_avg)}"
                        if len(period_avg)
                        else "0/0"
                    ),
                }
            )

    return pd.DataFrame(rows)


def _bootstrap_period_ci(
    frame: pd.DataFrame,
    feature: str,
    outcome: str,
    *,
    repeats: int = 250,
    seed: int = 42,
) -> tuple[float, float]:
    periods = sorted(frame["Period"].dropna().unique().tolist())
    if len(periods) < 6:
        return float("nan"), float("nan")

    rng = np.random.default_rng(seed)
    values = []

    for _ in range(int(repeats)):
        sampled = rng.choice(periods, size=len(periods), replace=True)
        parts = []
        for new_block_id, period in enumerate(sampled):
            part = frame[frame["Period"] == period][[feature, outcome]].copy()
            if not part.empty:
                part["_boot_block"] = new_block_id
                parts.append(part)

        if not parts:
            continue

        boot = pd.concat(parts, ignore_index=True)
        rho = _spearman(boot[feature], boot[outcome], min_n=20)
        if np.isfinite(rho):
            values.append(float(rho))

    if len(values) < 50:
        return float("nan"), float("nan")

    return (
        float(np.quantile(values, 0.005)),
        float(np.quantile(values, 0.995)),
    )


def _quartile_spread(
    frame: pd.DataFrame,
    feature: str,
    outcome: str,
) -> float:
    work = frame[[feature, outcome]].copy()
    work[feature] = pd.to_numeric(work[feature], errors="coerce")
    work[outcome] = pd.to_numeric(work[outcome], errors="coerce")
    work = work.dropna()

    if len(work) < 40 or work[feature].nunique() < 4:
        return float("nan")

    q1 = float(work[feature].quantile(0.25))
    q3 = float(work[feature].quantile(0.75))
    low = work[work[feature] <= q1][outcome]
    high = work[work[feature] >= q3][outcome]

    if low.empty or high.empty:
        return float("nan")
    return float(high.mean() - low.mean())


def feature_stability_v3(
    events: pd.DataFrame,
    *,
    outcome: str = "Gross 10m bps",
) -> tuple[pd.DataFrame, str]:
    """Screen features separately for LONG and SHORT events.

    Candidate status is deliberately strict:
    - effect magnitude threshold
    - 99% period-cluster bootstrap CI excludes zero
    - quartile spread agrees with correlation direction
    - consistency across individual periods
    - leave-one-period-out stability
    """
    rows = []
    available = [f for f in FEATURES_V3 if f in events.columns]

    for side in ["LONG", "SHORT"]:
        side_events = events[events["Side"].astype(str) == side].copy()
        if side_events.empty:
            continue

        for feature in available:
            work = side_events[["Period", "Symbol", feature, outcome]].copy()
            work[feature] = pd.to_numeric(work[feature], errors="coerce")
            work[outcome] = pd.to_numeric(work[outcome], errors="coerce")
            work = work.dropna()

            if len(work) < 60:
                continue

            overall = _spearman(work[feature], work[outcome], min_n=40)
            if not np.isfinite(overall) or overall == 0:
                continue

            effect_sign = np.sign(overall)
            spread = _quartile_spread(work, feature, outcome)
            ci_low, ci_high = _bootstrap_period_ci(work, feature, outcome)

            periods = sorted(work["Period"].unique().tolist())
            period_rhos = []
            for period in periods:
                part = work[work["Period"] == period]
                rho = _spearman(part[feature], part[outcome], min_n=6)
                if np.isfinite(rho) and rho != 0:
                    period_rhos.append(float(rho))

            same_periods = sum(
                1 for rho in period_rhos if np.sign(rho) == effect_sign
            )

            loo_rhos = []
            for period in periods:
                part = work[work["Period"] != period]
                rho = _spearman(part[feature], part[outcome], min_n=40)
                if np.isfinite(rho) and rho != 0:
                    loo_rhos.append(float(rho))

            same_loo = sum(
                1 for rho in loo_rhos if np.sign(rho) == effect_sign
            )

            ci_excludes_zero = (
                np.isfinite(ci_low)
                and np.isfinite(ci_high)
                and (
                    (ci_low > 0 and effect_sign > 0)
                    or (ci_high < 0 and effect_sign < 0)
                )
            )
            spread_agrees = (
                np.isfinite(spread)
                and np.sign(spread) == effect_sign
            )
            period_fraction = (
                same_periods / len(period_rhos)
                if period_rhos
                else 0.0
            )
            loo_fraction = same_loo / len(loo_rhos) if loo_rhos else 0.0

            candidate = (
                abs(float(overall)) >= 0.12
                and ci_excludes_zero
                and spread_agrees
                and abs(float(spread)) >= 12.0
                and len(period_rhos) >= 6
                and period_fraction >= 0.65
                and len(loo_rhos) >= 6
                and loo_fraction >= 0.80
            )

            rows.append(
                {
                    "Side": side,
                    "Feature": feature,
                    "Events": int(len(work)),
                    "Overall Spearman": float(overall),
                    "99% period-bootstrap CI low": ci_low,
                    "99% period-bootstrap CI high": ci_high,
                    "10m Q4-Q1 spread bps": spread,
                    "Same-sign periods": f"{same_periods}/{len(period_rhos)}",
                    "Leave-one-period same sign": f"{same_loo}/{len(loo_rhos)}",
                    "Candidate feature": "YES" if candidate else "NO",
                }
            )

    summary = pd.DataFrame(rows)
    if summary.empty:
        return summary, (
            "No LONG or SHORT feature had enough observations for the stricter v3 period-cluster stability screen."
        )

    summary["_rank"] = (
        summary["Candidate feature"].eq("YES").astype(int) * 10.0
        + summary["Overall Spearman"].abs()
    )
    summary = (
        summary.sort_values("_rank", ascending=False)
        .drop(columns=["_rank"])
        .reset_index(drop=True)
    )

    candidates = summary[summary["Candidate feature"] == "YES"]

    if candidates.empty:
        verdict = (
            "No LONG/SHORT feature passed the v3 period-cluster robustness screen with a 99% "
            "bootstrap interval. Do not create another historical holdout rule yet."
        )
    else:
        descriptions = []
        for _, row in candidates.head(3).iterrows():
            descriptions.append(
                f"{row['Side']} {row['Feature']} "
                f"(rho {float(row['Overall Spearman']):+.2f}, "
                f"99% CI [{float(row['99% period-bootstrap CI low']):+.2f}, "
                f"{float(row['99% period-bootstrap CI high']):+.2f}], "
                f"spread {float(row['10m Q4-Q1 spread bps']):+.1f} bps, "
                f"periods {row['Same-sign periods']})"
            )
        verdict = (
            "v3 found development candidates that survived the stricter side-specific "
            "period-cluster screen: "
            + "; ".join(descriptions)
            + ". They are candidates only, not validated trading edges."
        )

    return summary, verdict

