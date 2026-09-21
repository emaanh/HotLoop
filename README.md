# HotLoop

A benchmark for autonomous agents doing real GPU performance engineering: give an agent a PyTorch
reference and a real GPU, measure how well it independently produces a correct, faster
implementation. Tasks are **procedurally generated and certified on hardware so that the workload
regime changes what the right optimization is**. First study: how model quantization affects
this ability.

Status: planning (2026-09-20). No code yet.

## Docs

| File | What |
|---|---|
| [docs/DESIGN.md](docs/DESIGN.md) | Thesis, task formulation, generator, transfer matrices, architecture, MVP |
| [docs/EVAL.md](docs/EVAL.md) | Correctness, timing protocol, anti-cheat |
| [docs/ECOSYSTEM.md](docs/ECOSYSTEM.md) | Survey of related work and what we reuse |
| [docs/COMPILER_GAPS.md](docs/COMPILER_GAPS.md) | Tasks as located gaps in compiler capability; families to build next |
| [docs/QUANT_STUDY.md](docs/QUANT_STUDY.md) | Quantization study design, model + arms |
| [docs/PLAN.md](docs/PLAN.md) | Milestones, gates, kill criteria, risks |
| [docs/DECISIONS.md](docs/DECISIONS.md) | Append-only decision log |
| [docs/BUDGET.md](docs/BUDGET.md) | Cost breakdown, levers, staged approval |
| [docs/ACCESS.md](docs/ACCESS.md) | Accounts, keys, infra state, spend |
| [docs/LOG.md](docs/LOG.md) | Lab notebook |
