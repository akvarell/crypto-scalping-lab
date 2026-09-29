# RESEARCH_DECISIONS

## 1. Research goal
The objective is not to find a green backtest.
The objective is to find an economically interpretable, statistically stable candidate that survives realistic costs and chronological validation.

## 2. No endless fixed holdouts
Already viewed historical windows are inspected data.
Do not keep inventing new fixed holdouts after every development result.

A serious future candidate should move to purged chronological walk-forward.

## 3. Causal timing
Every feature must be available before entry.
5m information is only known after the 5m candle closes.
Execution must use the first realistically tradable price after known_time.

## 4. Warm-up
All 5m research calculations require at least 48h prehistory before an evaluation period.

## 5. Prior-only baselines
Rolling volume/activity baselines must use shift(1) before rolling.
The current bar cannot be part of its own baseline.

## 6. Trade and context universes
Trade universe: top-5 ranked symbols.
Context universe: top-15 ranked symbols.

## 7. Survivorship caveat
The current public Binance symbol master is CURRENT_LISTED_ONLY.
Do not claim survivorship-free historical universe reconstruction.

## 8. Costs
Primary stress:
- 12 bps round trip
- 20 bps round trip
- 30 bps round trip

Zero-cost profitability is not enough.

## 9. Statistical evidence
Use:
- median
- trimmed mean
- positive-period fraction
- period-level aggregation
- period-cluster bootstrap
- leave-one-period-out
- symbol concentration

Discovery screens should be conservative because many hypotheses are inspected.

## 10. Current strategic decision
v3.2 returned NO_TIMING_CANDIDATE.
Do NOT run purged walk-forward on the current four event families.

Next:
v3.3 Microstructure & Context Event Lab.

New work must be economically different from threshold tuning of the old event families.
