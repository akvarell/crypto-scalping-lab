---
name: replay-validator
description: Validate deterministic crypto trading replay against recorded live market data, time ordering, latency scheduling, boot boundaries and regression gates. Use when changing live/replay planners, event envelopes, WAL readers, validators or simulation timing; not for parameter tuning.
---

# Replay validator

## Reconstruct the contract
1. Locate the authoritative runtime spec, prior validator versions, current implementation, test fixtures and recorded raw envelope format. If v3/v4 contracts live only on the VPS, obtain them before certifying equivalence.
2. Create an explicit matrix of **every** old and new validation invariant. No silent deletion; capture name, source revision, test fixture, current status and reason for any change.
3. Known historic regression-sensitive gates include `replay_must_not_read_live_wal = true` and `fixed_l2_subscription = true`. Earlier discussion also flagged **four** dropped v3 checks: the remaining two must be recovered from the actual prior validator, never guessed or marked complete.

## Time and storage contract
- Within one `boot_id`, schedule receipt-driven live/replay events using `recv_monotonic_ns`. `recv_wall_ms` is UTC reporting metadata, not scheduling authority.
- A boot boundary is a hard simulation/session boundary. Do not carry positions, pending orders, queued actions or monotonic comparisons across a reboot.
- Require durable raw-envelope capture before any corresponding decision is allowed. A write failure blocks all decision flow until safe recovery.
- Verify environment fingerprint (Python version, dependency lock/`pip freeze`, build/config hashes) at initial start and after restart. Flag drift instead of silently mixing sessions.
- Enforce frozen data selection, event-time vs known-time causality, stable tie-breaking, no future reads, no replay reads from live WAL.
- Validate burn-in interval completes before S0 (evaluation start). Burn-in cannot generate counted signals/orders.

## Regression fixtures
Use a small immutable raw fixture and compare repeated replay runs for identical normalized event/order/position/PNL traces (hash + event-level diff). Add cases for equal timestamps, out-of-order inputs, raw-write failure, boot rollover, reconnect gap, stale book, delayed order ACK, and restart with dependency drift.
If live data has nondeterministic exchange/network effects, define exactly which normalized events can be compared; do not claim byte-for-byte live equivalence without proof.

## Exit criteria
Output a pass/fail/unknown matrix with v3→v4 coverage, fixture SHA, config SHA, timing assumptions, first mismatch offset and reproduction steps. **No live promotion** while a mandatory gate is missing, UNKNOWN, or failed. Do not edit running collectors as a side effect.
