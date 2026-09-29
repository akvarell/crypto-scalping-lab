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
    "5m volume acceleration",
    "5m close location",
    "Breadth acceleration 15m",
    "Relative extension / ATR",
    "Impulse age 5m bars",
]

HORIZONS = [3, 5, 10, 15]


def _safe_spearman(x: pd.Series, y: pd.Series) -> float:
    pair = pd.concat(
        [
            pd.to_numeric(x, errors="coerce"),
            pd.to_numeric(y, errors="coerce"),
        ],
        axis=1,
    ).dropna()

    if len(pair) < 12:
        return float("nan")
    if pair.iloc[:, 0].nunique() < 4 or pair.iloc[:, 1].nunique() < 4:
        return float("nan")

    return float(
        pair.iloc[:, 0].rank(method="average").corr(
            pair.iloc[:, 1].rank(method="average")
        )
    )


def _quartile_spread(frame: pd.DataFrame, feature: str, outcome: str) -> tuple[float, int, int]:
    work = frame[[feature, outcome]].copy()
    work[feature] = pd.to_numeric(work[feature], errors="coerce")
    work[outcome] = pd.to_numeric(work[outcome], errors="coerce")
    work = work.dropna()

    if len(work) < 20 or work[feature].nunique() < 4:
        return float("nan"), 0, 0

    q1 = float(work[feature].quantile(0.25))
    q3 = float(work[feature].quantile(0.75))
    low = work[work[feature] <= q1][outcome]
    high = work[work[feature] >= q3][outcome]

    if low.empty or high.empty:
        return float("nan"), len(low), len(high)

    return float(high.mean() - low.mean()), int(len(low)), int(len(high))


def analyze_continuation_exhaustion(
    development_details: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, str]:
    """Development-only feature study for continuation vs exhaustion.

    No holdout data is used here and no trading threshold is optimized.
    The lab measures whether pre-entry features have a stable monotonic
    relationship with forward returns across several fixed horizons.
    """
    if development_details is None or development_details.empty:
        raise ValueError("Development entry events are missing.")

    required = {"Tier", "Period", "Symbol", "Event time"}
    missing = required - set(development_details.columns)
    if missing:
        raise ValueError(f"Development details missing columns: {sorted(missing)}")

    base = development_details[
        development_details["Tier"].astype(str) == "Baseline LONG/BULL"
    ].copy()

    if base.empty:
        raise RuntimeError("No Baseline LONG/BULL events are available for v2.6.")

    base = base.drop_duplicates(
        subset=["Period", "Symbol", "Event time"],
        keep="first",
    ).reset_index(drop=True)

    available_features = [f for f in FEATURES if f in base.columns]
    if not available_features:
        raise RuntimeError(
            "v2.6 diagnostic features are absent. Rerun the Research Pipeline "
            "so development events are regenerated with the current code."
        )

    rows = []
    for feature in available_features:
        for horizon in HORIZONS:
            outcome = f"Gross {horizon}m bps"
            if outcome not in base.columns:
                continue

            rho = _safe_spearman(base[feature], base[outcome])
            spread, low_n, high_n = _quartile_spread(base, feature, outcome)

            rows.append(
                {
                    "Feature": feature,
                    "Horizon min": int(horizon),
                    "Events": int(
                        base[[feature, outcome]]
                        .apply(pd.to_numeric, errors="coerce")
                        .dropna()
                        .shape[0]
                    ),
                    "Spearman": rho,
                    "Q4 - Q1 gross spread bps": spread,
                    "Q1 events": low_n,
                    "Q4 events": high_n,
                }
            )

    relationships = pd.DataFrame(rows)
    if relationships.empty:
        raise RuntimeError("No v2.6 feature/return relationships could be calculated.")

    summary_rows = []
    for feature, group in relationships.groupby("Feature", sort=False):
        valid = group[np.isfinite(group["Spearman"])].copy()
        if valid.empty:
            continue

        signs = np.sign(valid["Spearman"].astype(float))
        positive = int((signs > 0).sum())
        negative = int((signs < 0).sum())
        dominant = max(positive, negative)
        direction = (
            "Higher → better forward return"
            if positive > negative
            else "Higher → worse forward return"
            if negative > positive
            else "Mixed"
        )

        row10 = group[group["Horizon min"] == 10]
        rho10 = float(row10["Spearman"].iloc[0]) if not row10.empty else float("nan")
        spread10 = (
            float(row10["Q4 - Q1 gross spread bps"].iloc[0])
            if not row10.empty
            else float("nan")
        )

        median_rho = float(valid["Spearman"].median())
        median_abs_rho = float(valid["Spearman"].abs().median())

        coherent = (
            dominant >= 3
            and median_abs_rho >= 0.10
            and np.isfinite(spread10)
            and abs(spread10) >= 8.0
        )

        summary_rows.append(
            {
                "Feature": feature,
                "Direction": direction,
                "Same-sign horizons": f"{dominant}/{len(valid)}",
                "Median Spearman": median_rho,
                "Median |Spearman|": median_abs_rho,
                "10m Spearman": rho10,
                "10m Q4-Q1 spread bps": spread10,
                "Coherent diagnostic": "YES" if coherent else "NO",
            }
        )

    feature_summary = pd.DataFrame(summary_rows)
    if feature_summary.empty:
        raise RuntimeError("v2.6 could not summarize the feature relationships.")

    feature_summary["_strength"] = feature_summary["Median |Spearman|"].fillna(0.0)
    feature_summary = (
        feature_summary.sort_values(
            ["Coherent diagnostic", "_strength"],
            ascending=[False, False],
        )
        .drop(columns=["_strength"])
        .reset_index(drop=True)
    )

    coherent = feature_summary[
        feature_summary["Coherent diagnostic"].astype(str) == "YES"
    ].head(3)

    if coherent.empty:
        verdict = (
            f"v2.6 examined {len(base)} development events across 3/5/10/15m horizons. "
            "None of the pre-entry continuation/exhaustion features showed a sufficiently "
            "coherent relationship across horizons. Do not invent a new threshold rule yet; "
            "the next research step should add a different explanatory feature family rather "
            "than retune volume thresholds."
        )
    else:
        pieces = []
        for _, row in coherent.iterrows():
            direction = (
                "better"
                if "better" in str(row["Direction"])
                else "worse"
            )
            pieces.append(
                f"{row['Feature']} → {direction} returns "
                f"({row['Same-sign horizons']} horizons, median rho "
                f"{float(row['Median Spearman']):+.2f}, 10m Q4-Q1 "
                f"{float(row['10m Q4-Q1 spread bps']):+.1f} bps)"
            )

        verdict = (
            f"v2.6 examined {len(base)} development events. The clearest development-only "
            "continuation/exhaustion relationships were: "
            + "; ".join(pieces)
            + ". These are hypothesis-discovery results only. Freeze one simple economic rule "
            "before looking at the third untouched historical window."
        )

    return feature_summary, relationships, verdict
