# RESEARCH_LOG

Append-only research journal.

## 2026-09-29 — v3.0 Research Reset

### Why
Earlier v2 research showed unstable development-only edges and exposed methodological weaknesses.

### Changes
- Added 48h 5m warm-up.
- Relative volume now uses prior candles only.
- Separated top-5 trade universe from top-15 context universe.
- Built broad causal breakout events instead of threshold-preselected signals.
- Added 1m causal execution timing.
- LONG/SHORT analyzed separately.
- Added 12/20/30 bps cost stress.
- Added period-cluster robustness.

### Result
6541 causal events / 12 periods / 28 symbols.

12 bps broad breakout baseline:
- net avg -10.9 bps
- net median -14.5 bps
- positive periods 1/12

Feature stability:
NO_CANDIDATE

### Conclusion
The clean breakout baseline has almost no gross edge and no robust feature survived the stricter period-cluster screen.

---

## 2026-09-29 — v3.1 Event Family Benchmark

### Hypothesis
A different event mechanism may have better unconditional gross expectancy than simple breakout continuation.

### Families
- Breakout continuation
- Failed breakout reversal
- Pullback reclaim
- Cross-sectional leader/laggard

### Result
22361 comparable events / 12 periods / 28 symbols.

Family counts:
- Cross-sectional leader/laggard: 9048
- Breakout continuation: 6611
- Pullback reclaim: 3568
- Failed breakout reversal: 3134

Status:
NO_FAMILY_CANDIDATE

Strongest descriptive group:
Failed breakout reversal / SHORT

At 12 bps:
- net avg -9.3 bps
- net median -10.7 bps
- positive periods 2/12

### Conclusion
No event family had enough gross edge to justify strategy validation.

---

## 2026-09-29 — v3.2 Execution & Horizon Map

### Hypothesis
The v3.1 event families might contain edge at a different execution delay or holding horizon.

### Tested
- 4 event families
- LONG / SHORT
- Immediate execution
- +1m delay
- 1 / 3 / 5 / 10 / 15 / 30 minute horizons
- 12 / 20 / 30 bps costs

### Result
45148 execution observations / 12 periods / 28 symbols.

Status:
NO_TIMING_CANDIDATE

Strongest descriptive combination:
Failed breakout reversal / SHORT / +1m delay / 10m

At 12 bps:
- net avg -7.8 bps
- net median -9.7 bps
- positive periods 2/12
- 99% cluster CI low -15.9 bps

### Conclusion
Execution timing and holding period do not rescue the current event families.
Do not tune the map.

### Next hypothesis
Search genuinely different causal microstructure/context mechanisms:
order-flow imbalance, breadth expansion/contraction, normalized cross-sectional dislocation, and microstructure reversal.


---

## 2026-09-29 — v3.3 Microstructure & Context Event Lab — PRE-RUN HYPOTHESIS

### Why
v3.2 showed that execution timing and holding horizon do not rescue the existing four event families. The next test must therefore change the economic mechanism rather than tune old thresholds.

### Hypothesis before observing v3.3 results
Short-horizon edge, if present, may be linked to causal microstructure/context state rather than breakout shape itself.

Four mechanism classes are tested:

1. Order-flow imbalance
   - taker-buy imbalance
   - change in taker imbalance
   - volume × imbalance interaction
   - average-trade-size × imbalance interaction

2. Breadth / participation
   - breadth expansion aligned with basket direction
   - simultaneous new highs/lows across context symbols
   - context-basket volume participation

3. Cross-sectional dislocation
   - relative return normalized by context-basket dispersion
   - continuation and mean-reversion orientations tested separately

4. Microstructure reversal
   - 5m flow reversal against the completed relative move
   - last-1m flow turn inside the completed 5m candle
   - close-location deterioration against the completed relative move

### Sampling / anti-selection design
Observations are sampled deterministically once per hour for each selected trade symbol.
Sampling does NOT depend on the magnitude of the feature being studied.
This prevents a magnitude threshold from being tuned before the relationship screen.

### Timing
All 5m/context features are known only after the completed 5m candle.
The modeled entry is the first available 1m open at/after known_time.
The 1m flow-turn feature uses only 1m bars contained inside the already completed 5m candle.

### Evaluation
- development/research data only
- LONG / SHORT separately
- horizons: 3 / 5 / 10 / 15 / 30 minutes
- continuous Strength -> signed future return Spearman
- Q4-Q1 effect size is descriptive
- 99% period-cluster bootstrap CI
- individual-period direction consistency
- leave-one-period-out stability
- symbol concentration check

### Promotion rule
v3.3 can only produce MECHANISM_CANDIDATE_FOUND, not a validated strategy.
No full-sample threshold may be frozen from v3.3.
If a mechanism survives, the next stage is v3.4 Purged Walk-Forward, where thresholds are derived inside each training fold only.
