from __future__ import annotations

"""Backward-compatible import shim for older Streamlit deployments.

The futures positioning implementation lives in src.futures_positioning_v332.
This module keeps the previous import path working without duplicating or changing
any strategy/research logic.
"""

from src.futures_positioning_v332 import (
    analyze_futures_positioning_v332,
    build_futures_positioning_period_v332,
    clear_futures_v332_caches,
    futures_source_coverage_v334,
)

__all__ = [
    "analyze_futures_positioning_v332",
    "build_futures_positioning_period_v332",
    "clear_futures_v332_caches",
    "futures_source_coverage_v334",
]
