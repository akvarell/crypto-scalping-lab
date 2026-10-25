# Crypto Scalping Lab skills

Repo-local skills for Codex. Each folder contains a discoverable `SKILL.md` with `name` and `description` YAML frontmatter; see root `AGENTS.md` for shared rules.

| Skill | Typical explicit invocation |
| --- | --- |
| market-data-auditor | `$market-data-auditor audit yesterday's L2 capture` |
| replay-validator | `$replay-validator compare the v3 and v4 replay gates` |
| backtest-auditor | `$backtest-auditor challenge the cost and timing assumptions` |
| orderbook-analyst | `$orderbook-analyst formalize the density-reposition hypothesis` |
| risk-manager | `$risk-manager review the paper-to-live controls` |
| vps-reliability | `$vps-reliability diagnose collector memory pressure` |
| universe-screener | `$universe-screener audit historical TOP-5 ranking` |

Codex can also select a matching skill based on the description. These are **instruction-only** skills: installing them changes agent workflows, not the trading bot, VPS, runtime configuration, order permissions, or systemd services.

## Maintenance
- Keep `name` equal to its directory basename; provide specific, non-overlapping triggering descriptions.
- Read the corresponding `SKILL.md` and current project docs before applying a skill.
- Verify mandatory validation checks against source and prior revisions when runtime is changed.
- Treat historical project details as versioned context, not as proof about running services.
