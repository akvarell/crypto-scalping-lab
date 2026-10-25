---
name: risk-manager
description: Design and audit fail-closed trading risk controls, order sizing, drawdown limits, kill switches, exchange credential boundaries and paper-to-live promotion. Use when editing order execution, position tracking or safety policies; not as approval to place live orders.
---

# Risk manager

## Default safety state
All new systems start OFFLINE/PAPER; real exchange order placement requires a separate explicit user authorization plus verified operational controls. This skill cannot itself approve a trade or a deployment.

## Build the risk envelope
- Specify and unit-test bounded per-symbol and portfolio exposure, leverage, max notional, position count, order frequency, maximum drawdown and daily loss budget. Values are explicit config choices, **not** silently invented defaults.
- Validate tick size, lot size, minimum order size/notional, available collateral and margin mode for each exchange and contract.
- Net positions across simultaneous strategies and pending orders; prevent duplicate intents, replayed commands and double execution after reconnect.
- Model stops, market gaps, liquidity holes, funding, liquidation distance, rate limits, exchange degradation and stale-quote failures.
- Enforce a single authoritative position/order state machine including partial fills, cancellations, expired intents, rejected orders, ACK latency and reconciliation with exchange balances.
- Place a hard reject/kill gate on unknown position state, stale/gapped L2, raw-capture failure, time/boot discontinuity, repeated order errors or breached loss limits.
- An emergency stop blocks **new** orders; define separately who may cancel resting orders or flatten positions, because these are real trade actions.

## Operational security
- Credentials only through properly scoped secrets (withdrawal permission disabled where supported); do not log keys or store them in GitHub.
- Split paper and production environments/accounts/config. Review permissions and explicit user approval before any production action.
- Keep a human-auditable trace of intent, pre-trade risk checks, order request, exchange response and post-trade reconciliation.

## Required simulation tests
High-frequency duplicate intents; simultaneous strategies; position/ACK mismatch; network partition; restart and boot change; stale book; one-sided liquidity; reject/partial fill; sudden volatility spike; raw-write failure; crossed book.
Report gate coverage and unknowns. No strategy edge, backtest or successful paper run is alone sufficient for live promotion.
