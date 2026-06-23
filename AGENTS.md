## Agent skills

### Issue tracker

GitHub Issues. External PRs are also treated as a triage surface. See `docs/agents/issue-tracker.md`.

### Triage labels

Default vocabulary (`needs-triage`, `needs-info`, `ready-for-agent`, `ready-for-human`, `wontfix`). See `docs/agents/triage-labels.md`.

### Domain docs

Single-context repo. See `docs/agents/domain.md`.

## Active issues

The PRD (`PRD.md`) has been broken into 7 tracer-bullet vertical slices, each labeled `ready-for-agent`:

| # | Title | Blocked by |
|---|-------|------------|
| #1 | Scaffold + core types | — |
| #2 | Config loading | #1 |
| #3 | Ollama API client | #1 |
| #4 | Metrics computation | #1 |
| #5 | Benchmark runner | #3, #4 |
| #6 | CLI: list + run commands | #2, #3, #5 |
| #7 | TUI: full flow | #2, #3, #5 |

Dependency order: #1 → #2/#3/#4 (parallel) → #5 → #6/#7 (parallel).
