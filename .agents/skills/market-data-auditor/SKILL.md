---
name: market-data-auditor
description: Audit crypto market-data ingestion, order-book L2 snapshots/deltas, trade tape, timestamps, raw WAL durability and dataset completeness. Use before consuming recorded streams for replay/backtests or when investigating gaps, disconnects, corrupted captures and mixed clocks; not for choosing trades.
---

# Market-data auditor

## First establish evidence
1. Identify the actual ingestion/runtime checkout, service, symbol set, venue, channel, schema and capture window. GitHub's `src/` historical research code is **not** evidence that VPS `scalping-v2` files match it.
2. Capture read-only source metadata: exchange, stream, snapshot mode/depth, fixed subscriptions, run/boot/session ID, UTC boundaries, code/config hash, compression and file manifest.
3. Never treat exchange event time, local receive wall time and receive monotonic time as interchangeable. Check all units and any reconnect/host-reboot boundary.

## Validate the raw feed before features
- Test field/types and schema version, unique event IDs where provided, increasing exchange sequence/update IDs, missing ranges, duplicates, out-of-order events, stalled streams and reconnect overlaps.
- For order books, verify the venue-specific snapshot + delta reconciliation contract, checksum when supported, crossed/negative books, stale best quotes, and recovery by resnapshot after a gap. A latest snapshot without continuity must not silently certify intervening data.
- Audit trade tape order and exchange timestamp reliability; do not fabricate microsecond precision from millisecond feeds.
- Check coverage by minute, symbol and stream; event-count spikes/zeros; disconnect and resync durations; freshness/latency distributions (p50/p95/p99) where clock origins permit.
- Confirm raw envelopes are durably persisted **before** use by the decision pipeline. On raw-write failures the decision path must fail closed. Do not infer fsync guarantees from a file merely existing.
- Validate .jsonl/.zst readability, checksums or stable content hashes, rotations, partial last records, monotonic segment ordering and disk usage; never mutate the source capture during the audit.

## Diagnostic tests
- Inject or simulate: sequence gap, duplicate delta, stale snapshot, disconnect, partial write, disk-full/raw-write error, exchange-time reversal, boot change, schema drift.
- Expect explicit reject/quarantine/resync plus counters, not silent record skipping.
- Preserve raw immutable input and produce derived/indexed outputs with source offsets and provenance.

## Report format
`PASS | FAIL | UNKNOWN` per invariant; evidence (filename/offset/time/run ID), counts, gap intervals, impact on symbols/strategies, quarantine recommendation, and smallest safe repair. UNKNOWN is not PASS. Do not claim complete continuity without exchange sequence or equivalent proof.
