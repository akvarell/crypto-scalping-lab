from __future__ import annotations

import numpy as np
import pandas as pd


FEATURES = [
    "1m confirm volume ratio",
    "1m confirm body bps",
    "5m volume ratio",
    "5m range / ATR",
    "5m relative move / ATR",
    "Breadth",
]

TIERS = [
    "Baseline LONG/BULL",
    "Strong 1m confirm",
    "Dual strong",
]


def _trimmed_mean(series: pd.Series) -> float:
    clean = pd.to_numeric(series, errors="coerce").dropna().astype(float)
    if clean.empty:
        return 0.0
    if len(clean) < 20:
        return float(clean.mean())
    lo = clean.quantile(0.10)
    hi = clean.quantile(0.90)
    trimmed = clean[(clean >= lo) & (clean <= hi)]
    return float(trimmed.mean()) if not trimmed.empty else float(clean.mean())


def _top_symbol_share(frame: pd.DataFrame) -> float:
    if frame.empty or "Symbol" not in frame:
        return 0.0
    counts = frame["Symbol"].value_counts()
    return float(counts.iloc[0] / counts.sum() * 100.0) if not counts.empty else 0.0


def _safe_spearman(x: pd.Series, y: pd.Series) -> float:
    """Spearman rank correlation without scipy.

    Spearman is simply Pearson correlation of the ranked observations.
    Using pandas rank() + ordinary Pearson avoids pulling scipy into the
    Streamlit environment just for this diagnostic.
    """
    pair = pd.concat(
        [
            pd.to_numeric(x, errors="coerce"),
            pd.to_numeric(y, errors="coerce"),
        ],
        axis=1,
    ).dropna()

    if len(pair) < 8:
        return float("nan")
    if pair.iloc[:, 0].nunique() < 3 or pair.iloc[:, 1].nunique() < 3:
        return float("nan")

    rank_x = pair.iloc[:, 0].rank(method="average")
    rank_y = pair.iloc[:, 1].rank(method="average")
    return float(rank_x.corr(rank_y))


def _robust_shift(dev: pd.Series, hold: pd.Series) -> float:
    d = pd.to_numeric(dev, errors="coerce").dropna().astype(float)
    h = pd.to_numeric(hold, errors="coerce").dropna().astype(float)
    if d.empty or h.empty:
        return float("nan")

    pooled = pd.concat([d, h], ignore_index=True)
    iqr = float(pooled.quantile(0.75) - pooled.quantile(0.25))
    if abs(iqr) < 1e-12:
        std = float(pooled.std())
        if abs(std) < 1e-12:
            return 0.0
        return float((h.median() - d.median()) / std)

    return float((h.median() - d.median()) / iqr)


def compare_feature_drift(
    development_details: pd.DataFrame,
    holdout_details: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Compare v2.2 development events with the frozen v2.3 holdout.

    This module is descriptive only: it does not create thresholds, rank
    candidate rules, or use holdout results to optimize a new strategy.
    """
    if development_details is None or development_details.empty:
        raise ValueError("Development event details are missing. Run v2.2 first.")
    if holdout_details is None or holdout_details.empty:
        raise ValueError("Frozen holdout event details are missing. Run v2.3 first.")

    dev = development_details.copy()
    hold = holdout_details.copy()
    dev["Sample"] = "Development"
    hold["Sample"] = "Holdout"

    common_tiers = [
        tier
        for tier in TIERS
        if tier in set(dev["Tier"].astype(str))
        and tier in set(hold["Tier"].astype(str))
    ]
    if not common_tiers:
        raise RuntimeError("No common entry-quality tiers exist in both samples.")

    edge_rows = []
    drift_rows = []
    corr_rows = []

    for tier in common_tiers:
        d = dev[dev["Tier"] == tier].copy()
        h = hold[hold["Tier"] == tier].copy()

        for sample_name, frame in [("Development", d), ("Holdout", h)]:
            gross = pd.to_numeric(frame["Gross bps"], errors="coerce").dropna()
            net = pd.to_numeric(frame["Net bps"], errors="coerce").dropna()

            edge_rows.append(
                {
                    "Tier": tier,
                    "Sample": sample_name,
                    "Events": int(len(frame)),
                    "Periods": int(frame["Period"].nunique()),
                    "Symbols": int(frame["Symbol"].nunique()) if "Symbol" in frame else 0,
                    "Top symbol share %": _top_symbol_share(frame),
                    "Gross avg bps": float(gross.mean()) if not gross.empty else 0.0,
                    "Gross median bps": float(gross.median()) if not gross.empty else 0.0,
                    "Trimmed gross avg bps": _trimmed_mean(gross),
                    "Net avg bps": float(net.mean()) if not net.empty else 0.0,
                    "Net median bps": float(net.median()) if not net.empty else 0.0,
                    "Net positive events %": (
                        float((net > 0).mean() * 100.0)
                        if not net.empty
                        else 0.0
                    ),
                }
            )

        for feature in FEATURES:
            if feature not in d.columns or feature not in h.columns:
                continue

            dvals = pd.to_numeric(d[feature], errors="coerce").dropna()
            hvals = pd.to_numeric(h[feature], errors="coerce").dropna()
            if dvals.empty or hvals.empty:
                continue

            drift_rows.append(
                {
                    "Tier": tier,
                    "Feature": feature,
                    "Development median": float(dvals.median()),
                    "Holdout median": float(hvals.median()),
                    "Median delta": float(hvals.median() - dvals.median()),
                    "Development IQR": float(dvals.quantile(0.75) - dvals.quantile(0.25)),
                    "Holdout IQR": float(hvals.quantile(0.75) - hvals.quantile(0.25)),
                    "Robust shift": _robust_shift(dvals, hvals),
                }
            )

            d_corr = _safe_spearman(d[feature], d["Gross bps"])
            h_corr = _safe_spearman(h[feature], h["Gross bps"])
            same_sign = (
                bool(np.sign(d_corr) == np.sign(h_corr))
                if np.isfinite(d_corr) and np.isfinite(h_corr)
                else False
            )

            corr_rows.append(
                {
                    "Tier": tier,
                    "Feature": feature,
                    "Development Spearman": d_corr,
                    "Holdout Spearman": h_corr,
                    "Correlation delta": (
                        float(h_corr - d_corr)
                        if np.isfinite(d_corr) and np.isfinite(h_corr)
                        else float("nan")
                    ),
                    "Same direction": "YES" if same_sign else "NO",
                }
            )

    edge = pd.DataFrame(edge_rows)
    drift = pd.DataFrame(drift_rows)
    corr = pd.DataFrame(corr_rows)

    return edge, drift, corr
