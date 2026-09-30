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


---

## 2026-09-29 — v3.3 Microstructure & Context Event Lab — RESULT

### Dataset
94526 continuous mechanism observations
12 periods
28 symbols

Category observations:
- Order flow: 39972
- Breadth: 21331
- Dislocation: 20140
- Microstructure reversal: 13083

### Result
NO_MECHANISM_CANDIDATE

No relationship passed the full predeclared discovery gate.

Strongest descriptive relationship:
- Relative-z reversion
- SHORT
- 15m
- Spearman strength→return: +0.07
- Q4-Q1 gross spread: +15.9 bps
- Q4 gross median: +13.2 bps
- positive periods: 8/12
- 99% spread CI low: +2.0 bps

### Interpretation
This relationship did not pass because the predeclared gate required stronger overall monotonic association, including rho >= 0.08. The rho gate must NOT be relaxed after seeing the result.

However, the direction, tail effect, period breadth and positive 99% spread CI justify a replication study on already-inspected historical eras before deciding whether the mechanism deserves purged walk-forward.

---

## 2026-09-29 — v3.3.1 Cross-Era Relative-z Replication — PRE-RUN HYPOTHESIS

### Frozen descriptive mechanism
Positive cross-sectional relative-z dislocation may mean-revert downward over the next 15 minutes.

Exact orientation:
- relative_z > 0 implies the coin has outperformed the context basket relative to basket dispersion
- test SHORT reversion only
- strength = positive relative_z
- horizon = 15m
- deterministic hourly sampling
- immediate next-tradable 1m execution after the completed 5m candle

### Replication eras
Both are already inspected by prior research and therefore are NOT holdouts:
- Era A: 90d ending 270d before the frozen anchor
- Era B: 90d ending 390d before the frozen anchor

No new untouched historical window will be opened.

### Replication evidence
For each era and pooled older data, report:
- events / periods / symbols
- Spearman strength→15m short return
- Q4-Q1 gross spread
- Q4 gross median
- Q4 trimmed gross
- positive periods
- top-symbol share
- 99% period-cluster spread CI

### Replication gate
Call REPLICATED only if:
- both eras have positive Spearman
- both eras have positive Q4-Q1 spread
- both eras have positive Q4 median
- both eras have >50% positive Q4 periods
- pooled older sample has positive Q4 median
- pooled older sample has 99% period-cluster spread CI low > 0
- pooled Q4 top-symbol share <= 35%

Even REPLICATED is not strategy validation because both eras are already inspected.
A replicated mechanism may advance only to purged chronological walk-forward.


---

## 2026-09-29 — v3.3.1 Cross-Era Relative-z Replication — RESULT

### Frozen mechanism
Positive relative-z dislocation -> SHORT reversion, fixed 15m horizon.
No threshold, side or horizon changes were allowed after v3.3.

### Result
Status: NOT_REPLICATED

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

Pooled:
- 10322 events / 24 periods / 49 symbols
- rho +0.05
- Q4-Q1 +0.5 bps
- Q4 median +9.5 bps
- positive periods 17/24
- 99% spread CI low -11.6 bps

### Conclusion
The exact v3.3 relative-z near-miss does not replicate cleanly across older eras.
Reject it without changing side, horizon or threshold definition.

---

## 2026-09-29 — v3.3.2 Futures Positioning & Basis Lab — PRE-RUN HYPOTHESIS

### Why
Spot price-shape, timing, order-flow, breadth and cross-sectional dislocation mechanisms have not produced a stable edge.
A genuinely different information source is required before declaring NO_EDGE_FOUND.

### Economic hypothesis
Short-horizon spot returns may contain information from perpetual-futures crowding:
- premium/index dislocation may reflect leveraged directional demand;
- extreme or changing premium may either continue briefly or mean-revert;
- funding sign/magnitude can proxy persistent positioning pressure;
- futures-vs-spot basis changes may reveal temporary crowding/dislocation.

### Predeclared mechanism orientations
Both continuation and reversion are allowed only where they represent distinct economic hypotheses and are labeled separately before results.

### Data/timing
- Binance USD-M public funding history
- Binance USD-M premium-index klines
- Binance USD-M futures klines where available
- only completed records at or before known_time
- funding record must satisfy fundingTime <= known_time
- deterministic hourly sampling
- spot outcome entry uses the first tradable 1m open after known_time

Historical open-interest is excluded because the public historical endpoint is limited to recent data and cannot support the frozen historical window consistently.

### Evaluation
Continuous strength relationships only.
No threshold optimization.
Horizons: 3 / 5 / 10 / 15 / 30m.
Strict period-cluster robustness before any purged walk-forward.


---

## 2026-09-30 — v3.3.2 Futures Lab — TECHNICAL FAILURE

### Observed failure
The Streamlit run returned zero futures-positioning observations.

### Root cause
The implementation coupled all futures sources inside one required fetch path.
A failure of funding or another futures REST endpoint caused the entire symbol to be skipped.
This was a technical implementation error, not a negative research result.

### Fix in v3.3.3
- funding is optional and independent;
- premium, futures basis and funding can contribute separately;
- USD-M futures klines and premium-index klines now fall back to Binance's public data archive when REST returns no data or is unavailable;
- the research hypothesis and statistical gate are unchanged.

The v3.3.2 zero-observation run must not be interpreted as NO_EDGE_FOUND.


---

## 2026-09-30 — v3.3.3 Futures Lab — SECOND TECHNICAL FAILURE

### Observed failure
The rerun again returned zero futures-positioning observations.

### Additional root cause found
Binance uses different naming between REST and the public archive:
- REST endpoint: premiumIndexKlines
- public archive dataset: premiumPriceKlines

The fallback incorrectly used the REST name as the archive directory name.

### v3.3.4 fix
- corrected archive dataset name to premiumPriceKlines;
- added markPriceKlines + indexPriceKlines fallback so premium can be reconstructed independently;
- kept futures klines, premium and funding independent;
- added a source-coverage diagnostic that reports how many selected symbols actually returned:
  futures bars / premium bars / mark+index / funding.

No research gate, mechanism orientation or statistical threshold was changed.
Zero-observation runs remain technical failures and are not evidence of no edge.
