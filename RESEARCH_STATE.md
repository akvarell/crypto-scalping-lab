# RESEARCH_STATE

## Current version
v3.3 Microstructure & Context Event Lab

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
v3.3 Microstructure & Context Event Lab

Implementation is ready and awaiting the development-only run.

Mechanism families:
1. Order-flow imbalance
2. Breadth expansion / participation
3. Cross-sectional dislocation
4. Microstructure reversal

Design:
- deterministic hourly sampling, independent of feature magnitude
- no magnitude threshold used to select observations
- causal 5m features known at candle close
- immediate next-tradable 1m entry
- continuous strength -> future signed return analysis
- horizons: 3 / 5 / 10 / 15 / 30m
- LONG / SHORT separated
- 99% period-cluster spread CI
- leave-one-period stability
- quartile effect sizes are descriptive only

Do NOT open another historical holdout in v3.3.

## Target next validation
Only a robust new candidate may advance to v3.4 Purged Walk-Forward.

## Latest repository commit
17c9ff15c7af7bac0ddd19b646025fee9c21c3b5
