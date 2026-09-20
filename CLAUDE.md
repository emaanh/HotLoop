# HotLoop

Benchmark for autonomous agents doing real GPU performance engineering, with procedurally
generated regime-sensitive tasks, plus a study of how model quantization affects agent ability.

## Working rules

- **Docs are the source of truth and must stay current.** Whenever a decision is made, changed,
  or invalidated, update the relevant file in `docs/` *in the same change*:
  - `docs/DECISIONS.md` — append-only decision log (ADR style: context, decision, why, status).
    Never silently rewrite history; supersede with a new entry.
  - `docs/PLAN.md` — milestones, gates, current status. Update status as work lands.
  - `docs/LOG.md` — dated lab notebook: experiments run, numbers observed, surprises, dead ends.
  - `docs/ACCESS.md` — accounts/keys/infra state.
  - Topic docs (`DESIGN.md`, `EVAL.md`, `QUANT_STUDY.md`, `ECOSYSTEM.md`) hold the
    current design; keep them consistent with DECISIONS.
- The four components stay decoupled: `taskgen` (emits static task packages), `evaluator`
  (task package + submission dir → result JSON), `runner` (sandboxes, budgets, trajectories),
  `agents` (adapters; a model is just an OpenAI-compatible URL). No imports across these except
  through the on-disk task-package / submission / result schemas.
- Real hardware, objective measurements. Never report a number without the measurement
  conditions (GPU, driver, clocks, n, CI). Negative results go in `docs/LOG.md` too.
- Don't rebuild what exists. Check `docs/ECOSYSTEM.md` before writing infrastructure.
- Python tooling: `uv`. No secrets in the repo; keys live in `.env` (gitignored) / provider CLIs.
