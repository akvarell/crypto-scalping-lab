# Crypto Scalping Lab — repository agent rules

These rules apply to the entire repository and to any Codex session started inside it.

## Mission and source of truth
- Build a reproducible, causal, cost-aware research system; do not optimize for a good-looking backtest.
- Before modifying research logic read `RESEARCH_DECISIONS.md`, `RESEARCH_STATE.md`, and the affected source files. The documents represent the historical research state of this GitHub branch, **not** proof that later VPS runtime work is synchronized.
- Research code in this repository and the independently deployed scalping-v2 ingestion/runtime on the VPS may differ. Never claim a VPS fix, service status, dataset schema, or live/replay guarantee was verified from GitHub alone.
- Preserve previously rejected hypotheses and inspected-data labels. Do not recycle inspected windows as unseen holdouts.
- Record the data snapshot, code SHA, configuration, environment, UTC time boundaries, universe, costs, and assumptions for every reproducible experiment.

## Non-negotiable research invariants
1. No look-ahead: every feature has an explicit known time; enter no earlier than the first realistically tradable event strictly after that time.
2. Use causal historical selection, completed bars, prior-only rolling baselines, and appropriate 5m warm-up (at least 48 h for existing 5m research).
3. Respect research-versus-validation separation. In this branch, historical Binance symbols are CURRENT_LISTED_ONLY; never call that universe survivorship-free.
4. Include realistic fees, spreads, slippage, latency, and fill/queue assumptions. Preserve 12/20/30 bps round-trip research stress as a comparison baseline, not a universal live cost model.
5. Report side-separated, period-level, and symbol-concentration diagnostics; prefer chronological purged walk-forward for qualified new candidates.
6. Favor fail-closed behavior when market data, replay integrity, time ordering, or risk state is uncertain. Never manufacture missing data or results.

## Operational boundaries
- Default to read-only inspection, offline analysis, and paper simulation.
- Never place a real order, enable live trading, access private exchange credentials, change production service limits, restart/stop collectors, purge raw data, or deploy to the VPS without a separate explicit user request and a change/safety plan.
- Do not expose API keys, exchange credentials, private VPS secrets, or raw sensitive logs in commits or chat.
- Market data collection must be independently validated before results are used as evidence of a strategy edge.
- A skills file is guidance, **not** an executable protection; production guards must also exist in code/config and be tested.
- When changing validators, preserve **all** previously mandated checks. Compare against authoritative prior revision/contract first; a vanished check is a regression, even if current tests pass.

## Workflow for changes
1. State the task and identify impacted modules, code version, and whether the target is GitHub, VPS runtime, or both.
2. Choose the matching skill in `.agents/skills/`. Read the full `SKILL.md` before acting.
3. Make the smallest reviewable change. Write focused regression tests when touching behavior.
4. Run checks appropriate to the changed modules; if they cannot be run, say exactly which were not run.
5. Report: changed paths, tests/evidence, unresolved risks, what is deployed versus merely proposed.
6. Prefer an isolated branch and reviewable pull request over unreviewed changes to `main`.

## Available repository skills
- `market-data-auditor`: ingestion/L2/raw-capture integrity
- `replay-validator`: live/replay determinism, causal clocks, regression gates
- `backtest-auditor`: causal simulation, costs, validation, robustness
- `orderbook-analyst`: tape/L2 hypotheses, spoofing and liquidity formations
- `risk-manager`: limits, kill switches, safe paper-to-live gates
- `vps-reliability`: Linux/systemd, OOM, storage and service safety
- `universe-screener`: causal asset selection and rolling universe

Invoke via `$skill-name` in Codex or describe the task. Skills help review and author code; they do not independently monitor the VPS.
