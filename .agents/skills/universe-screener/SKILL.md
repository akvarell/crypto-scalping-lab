---
name: universe-screener
description: Audit and evolve the crypto Rolling Universe screener: historical causal candidate selection, liquidity/volatility/ATR/activity ranking, BTC exclusion, diversification and forward windows. Use when choosing trade/context baskets or comparing screener revisions; not when optimizing trade entries.
---

# Universe screener

## Canonical historical baseline (verify against code before change)
Existing source: `src/rolling_universe.py`, `src/universe.py`, `src/research_universe_v3.py` plus `RESEARCH_DECISIONS.md`.
- Frozen research anchor: `2026-09-29 00:00 UTC`. This is a reproducibility anchor for past research, **not** a present-day live selection clock.
- Baseline: 14-day lookback, roughly 20 liquid candidate symbols, top-5 trade basket, top-15 context basket (as used in later research), 7-day forward evaluation, minimum average daily USDT turnover around $5m.
- Score is percentile ranks within the *past-only candidate pool*: `0.35 * realized_vol_rank + 0.30 * liquidity_rank + 0.25 * ATR_rank + 0.10 * activity_rank`. Define metric units, ranking tie rules and missing-value behavior.
- User preference: exclude BTC from **trade candidates**. Confirm an explicit BTC filter exists before claiming compliance; the inspected `rolling_universe.py` historically filters stable/base asset types but does not explicitly reject BTC.
- Historical master is CURRENT_LISTED_ONLY and thus survivorship-biased. Do not present as reconstructed historical listings.

## Validation
1. Build candidate pools from completed past-only bars available strictly before each selection. Prevent future 24h ticker turnover, current exchangeInfo changes, future listed status or future data leakage from entering ranking.
2. Verify spot/perpetual venue contracts and consistent volume units; separate quote turnover, trade count/activity, realized vol and ATR; no mix of inconsistent market types.
3. Apply min-history/new listing rules and report excluded symbols and coverage; use causal universe membership with deterministic ties.
4. Validate scores on 14d lookback and opportunity on *next* 7d without adapting past selections to the observed winners. Record turnover coverage, pool size, selection turnover and ranking drift.
5. Audit concentration, correlations/cross-asset redundancy, liquidity gaps, newly listed assets and regime sensitivity. Correlation diversification is a supplementary observable, not an invented fixed weight in the baseline formula.
6. Report robustness across chronological folds, insufficient data, delistings, exchange failures and market regime changes; never optimize the same forward window post hoc.

## Output
Deliver selection timestamp, available symbols, exclusions, input hashes, normalization/ranks, trade TOP-5, context TOP-15, fallback policy and confidence/limitations. Separate RESEARCH, PAPER LIVE SELECTION and ACTUAL EXECUTION; never send orders from this skill.
