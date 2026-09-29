# RESEARCH_STATE

## Current version
v3.3.1 Cross-Era Relative-z Replication

## Current status
READY_TO_RUN

## Last completed test
Execution & Horizon Map v3.2

## Last result
- 45,148 execution observations
- 12 periods
- 28 symbols
- 4 event families
- LONG / SHORT
- Immediate / +1m delay
- Horizons: 1 / 3 / 5 / 10 / 15 / 30 minutes
- Costs: 12 / 20 / 30 bps
- No timing/execution/horizon combination passed the strict 12 bps candidate screen.

Strongest descriptive combination:
- Failed breakout reversal
- SHORT
- +1m delay
- 10m hold
- Net avg: -7.8 bps
- Net median: -9.7 bps
- Positive periods: 2/12
- 99% cluster CI low: -15.9 bps

This is diagnostic only and is NOT an edge.

## What is rejected
- Plain breakout continuation as a broad edge
- Strong1m / DualStrong entry-quality rules
- Simple BTC-relative context
- Alt-basket continuation rules tested in v2
- Frozen continuation v2.7
- Frozen low-close-location v2.9 candidate path
- The four v3.1 event families as currently defined
- Execution timing / holding-horizon rescue of those same event families

## Inspected data
All historical windows used through v2.x are inspected and must not be called untouched again.

The 90-day v3 development/research window anchored at:
2026-09-29T00:00:00+00:00

is development/research data.

## Current data layer
- Frozen research anchor
- Trade universe: top-5
- Context universe: top-15
- 48h 5m warm-up
- Prior-only rolling volume baselines
- Causal event timing
- 1m execution
- LONG / SHORT analyzed separately
- Cost stress: 12 / 20 / 30 bps
- Period-cluster robustness
- Checkpointed period-by-period Streamlit jobs

## Known limitations
Historical Binance universe is CURRENT_LISTED_ONLY.
Delisted historical symbols are not reconstructed by the standard public endpoint.
Historical universe is therefore not fully survivorship-free.

## Next research step
v3.3.1 Cross-Era Relative-z Replication

v3.3 completed with NO_MECHANISM_CANDIDATE, but its strongest descriptive relationship was:
- Relative-z reversion
- SHORT
- 15m
- rho +0.07
- Q4-Q1 +15.9 bps
- Q4 median +13.2 bps
- positive periods 8/12
- 99% spread CI low +2.0 bps

The predeclared rho >= 0.08 gate was not met, so this is NOT a candidate and the gate must not be relaxed post-hoc.

Replication plan:
- freeze the exact mechanism orientation: positive relative-z dislocation -> SHORT reversion
- fixed 15m horizon
- same hourly sampling and immediate causal 1m execution
- no threshold optimization
- evaluate on two older, already-inspected eras
- Era A: 90d window ending 270 days before anchor
- Era B: 90d window ending 390 days before anchor
- no new untouched window is consumed

## Target next validation
Only a robust new candidate may advance to v3.4 Purged Walk-Forward.

## Latest research code commit
17c9ff15c7af7bac0ddd19b646025fee9c21c3b5
