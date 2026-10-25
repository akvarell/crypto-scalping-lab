---
name: backtest-auditor
description: Review crypto strategy backtests, walk-forward experiments and profitability reports for causal leakage, realistic execution/fees/slippage, period dependence and overfitting. Use when changing backtest engines or judging research results; not to assert that profit is guaranteed.
---

# Backtest auditor

## Locate the experimental contract
Read `RESEARCH_DECISIONS.md`, `RESEARCH_STATE.md`, `src/rolling_universe.py`, `src/backtest.py`, `src/execution_backtest.py`, `src/validation_v3.py` and the actual affected modules. Documents in main may describe the September research phase and not the newer VPS runtime.
Freeze and record: commit, input hashes, run configuration, evaluation UTC boundaries, symbol selection, strategy variant, cost model and exclusions.

## Causality checks
- Every selected symbol and derived feature must use only data available by its `known_time`.
- Completed 5m features are usable only after close; the 1m execution timeline must begin after the feature became available. Enforce correct next tradable price.
- Prior-only rolling baselines (e.g. `shift(1)`), no labels in features, no future-day liquidity for ranking, no forward-filled future observations.
- Use at least 48h prehistory for 5m research warm-up and do not count warm-up as evaluated trades.
- Universe in the existing historical Binance pipeline is CURRENT_LISTED_ONLY; disclose survivorship bias, delisting gaps and stale/new-listing sample limitations.
- Reused historical windows are inspected research data, not untouched out-of-sample; don't optimize after inspecting the final holdout.

## Execution and statistics
- Test LONG and SHORT separately; apply entry/exit fees, realistic bid-ask spread, slippage, taker/maker assumptions, funding when applicable, queue/fill probability, partial fills, latency and cancellations.
- Retain 12/20/30 bps round-trip cost stress as research comparators; don't substitute them for live exchange cost estimates.
- For intrabar stop/target ambiguity use a documented conservative order (the existing strict engine prioritizes stop); never count fills at unavailable quotes.
- Report gross and net, equity/drawdown, PF, median/trimmed mean, positive periods, period-cluster bootstrap confidence, leave-one-period-out and symbol concentration.
- Use purged chronological walk-forward (with embargo where labels overlap) only for a separately qualified candidate. Do not lower predeclared selection thresholds because a near-miss looked attractive.

## Adversarial tests and decision
Probe one-bar shift, look-ahead injection, fee doubling, delayed entry, spread widening, zero fills, missing candles, deterministic reruns, and remove-one-symbol/period stress.
Classify `INVALID_DATA | INVALID_EXECUTION | RESEARCH_ONLY | NEEDS_FORWARD_TEST | CANDIDATE` with explicit rationale. Even CANDIDATE is not permission to trade real funds.
