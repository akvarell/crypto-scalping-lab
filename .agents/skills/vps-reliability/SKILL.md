---
name: vps-reliability
description: Diagnose Linux VPS systemd collector health, cgroup memory/OOM events, disk capacity, logs, streaming reconnects and data retention. Use for ingestion stalls, crashes, resource limits or operational runbooks; inspect read-only first and never disrupt capture without approval.
---

# VPS reliability

## Identify the host and unit, do not assume parity
The deployment may be separate from `akvarell/crypto-scalping-lab` on GitHub. Confirm hostname, checkout SHA, exact unit name, mount points and data directories from the user's environment. Historic examples include `scalping-v2-ingest.service` and `/var/lib/scalping-v2`; treat them as examples until confirmed.

## Safe read-only triage
Use only explicitly supplied/authorized shell access; otherwise provide commands for the user to run:
```bash
systemctl show scalping-v2-ingest.service -p ActiveState -p SubState -p MainPID -p NRestarts -p MemoryCurrent -p MemoryMax
systemctl status scalping-v2-ingest.service --no-pager
journalctl -u scalping-v2-ingest.service -n 120 --no-pager
df -h /
du -sh /var/lib/scalping-v2
```
- For cgroup v2, read `memory.current`, `memory.max`, `memory.events`, `memory.peak` when available; compare `max`, `oom`, `oom_kill` **deltas** between samples rather than interpreting cumulative counters as new failures.
- Inspect CPU, RSS, disk bytes/inodes, filesystem permissions, I/O pressure, WAL rotations, clock synchronization, process restarts and network reconnect/backoff.
- Check for monotonic data growth and event freshness, not just a green `systemctl` state. A running collector can write stale or unusable records.
- Distinguish kernel OOM, systemd memory cgroup limits, process exit, blocked I/O, rate limit and exchange disconnection.

## Protection rules
Do NOT restart/stop services, raise resource limits, clear journals, vacuum raw captures, rotate/delete datasets, patch live binaries, modify systemd units, or change storage location without explicit authorization, rollback plan and capture-integrity review.
When `MemoryMax` is tight, propose bounded queues, streaming compression, chunk sizes and backpressure as code-level options; never drop events silently to meet memory limits.
Raw-write failure must fail closed upstream of decisions; disk-full recovery requires integrity checks and validated resync.

## Output
Summarize `CURRENT | TREND | RISK | SAFE NEXT STEP` with sampled times, exact command evidence, missing diagnostics and whether data capture stayed complete. Never claim continuous monitoring from a one-time check.
