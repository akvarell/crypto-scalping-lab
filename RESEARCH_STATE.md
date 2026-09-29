# RESEARCH_STATE

## Current version
v3.3.1 Cross-Era Relative-z Replication

## Current status
READY_TO_RUN

## Last completed test
Microstructure & Context Event Lab v3.3

## Last result
- 94,526 continuous mechanism observations
- 12 periods
- 28 symbols
- Order flow: 39,972 observations
- Breadth: 21,331
- Dislocation: 20,140
- Microstructure reversal: 13,083
- Status: NO_MECHANISM_CANDIDATE

Strongest descriptive relationship:
- Relative-z reversion
- SHORT
- 15m
- Spearman: +0.07
- Q4-Q1 gross spread: +15.9 bps
- Q4 gross median: +13.2 bps
- Positive periods: 8/12
- 99% spread CI low: +2.0 bps

The predeclared rho >= 0.08 gate was not met.
This is diagnostic evidence only, not a validated edge.

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
c4d60e2833caf61aef7643f070c36e88a39be5b4
