# RESEARCH_STATE

## Current version
v3.3.3 Futures Positioning & Basis Lab

## Current status
READY_TO_RERUN_AFTER_TECHNICAL_FIX

## Last completed test
Cross-Era Relative-z Replication v3.3.1

## Last result
v3.3.1 Cross-Era Relative-z Replication: NOT_REPLICATED

Era A:
- 4915 events / 12 periods / 26 symbols
- rho +0.04
- Q4-Q1 -6.8 bps
- Q4 median +8.3 bps
- positive periods 8/12
- 99% spread CI low -28.6 bps

Era B:
- 5407 events / 12 periods / 25 symbols
- rho +0.06
- Q4-Q1 +7.0 bps
- Q4 median +11.2 bps
- positive periods 10/12
- 99% spread CI low +2.3 bps

Pooled older eras:
- 10322 events / 24 periods / 49 symbols
- rho +0.05
- Q4-Q1 +0.5 bps
- Q4 median +9.5 bps
- positive periods 17/24
- 99% spread CI low -11.6 bps

The exact relative-z SHORT 15m near-miss did not replicate cleanly and is rejected without retuning.

## What is rejected
- Plain breakout continuation as a broad edge
- Strong1m / DualStrong entry-quality rules
- Simple BTC-relative context
- Alt-basket continuation rules tested in v2
- Frozen continuation v2.7
- Frozen low-close-location v2.9 candidate path
- The four v3.1 event families as currently defined
- Execution timing / holding-horizon rescue of those same event families
- Relative-z reversion / SHORT / 15m cross-era replication

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
v3.3.2 Futures Positioning & Basis Lab

This is a genuinely different data source/mechanism, not a retune of the failed spot-event families.

Pre-run mechanism families:
1. Perpetual premium crowding / reversion
2. Perpetual premium continuation
3. Premium change / shock
4. Funding crowding / reversion
5. Funding change
6. Futures-vs-spot basis/dislocation where historical futures klines exist

Design:
- development-only 90-day v3 research window
- same frozen spot trade universe
- hourly deterministic sampling
- only completed futures/spot bars available by known_time
- funding uses only records with fundingTime <= known_time
- no open-interest history because Binance historical OI endpoint is limited to recent data and is not comparable across the frozen research window
- no magnitude threshold optimization
- continuous Strength -> future spot return
- 3 / 5 / 10 / 15 / 30m horizons
- LONG / SHORT separately
- strict period-cluster robustness

## Target next validation
Only a robust new candidate may advance to v3.4 Purged Walk-Forward.

## Latest research code commit
001d7516ac21bccedd9c4feb060306d1a022fad1
