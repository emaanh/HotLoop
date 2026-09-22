# HotLoop

A benchmark for autonomous agents doing real GPU performance engineering: give an agent a PyTorch
reference and a real GPU, measure how well it independently produces a correct, faster
implementation. Tasks are **procedurally generated and certified on hardware so that the workload
regime changes what the right optimization is**.

**Status (2026-09-21): working end to end on rented A100s.** 12 certified tasks, a trusted
evaluator with an anti-cheat regression suite, a sandbox runner, an agent adapter, and first
results from a frontier model. Not yet done: a second model, the cross-regime analysis of agent
solutions, roofline-relative scoring, and the quantization study (parked).

## What exists

| | |
|---|---|
| `benchmark/dev-v0/` | **12 sealed task packages** — 6 with 3.5–18× certified headroom over `torch.compile`, 2 diagnostic, 4 no-headroom controls |
| `packages/schemas` | the on-disk contracts every other component shares (task package, `result.json`, `trajectory.jsonl`) |
| `packages/taskgen` | task families + emission + certification against the real evaluator |
| `packages/evaluator` | correctness (fp64-calibrated tolerances), paired-block timing with bootstrap CIs, anti-cheat |
| `packages/runner` | GPU VM lifecycle, sandbox containers, trusted scoring containers |
| `packages/agents` | agent adapter over [mini-swe-agent](https://github.com/SWE-agent/mini-swe-agent) 2.4.6 |
| `image/` | the pinned CUDA 12.8 / torch 2.11 / Triton 3.6 image agents and the evaluator run in |
| tests | 82 passing, 6 GPU-only (skipped without CUDA) |

## Results so far

**Measurement** (A100-SXM4-40GB, clocks locked): raw block CV 0.1–1.6%; A-vs-A paired ratio
1.0000; a planted 1.10× is recovered as 1.099–1.100 and detected >95% of the time. Between fresh
evaluations of the same solution the spread is ~2%, so scores of record pool ≥3 runs.

**Agent** (`gpt-5.5`, 13 clean trajectories, 3 hidden-seed evaluations each, no successful cheats):

| task | certified best-known | best lazy wrapper | gpt-5.5 |
|---|---|---|---|
| ragged pooling, tiny segments + outliers | 3.48× | 1.00× | **17.3×** |
| batched chain, vector on the right | 18.13× | 1.01× | **22.4×** |
| ragged pooling, heavy tail | 10.93× | 1.00× | **11.6×** |
| control: chain already optimal | 1.00× | 1.00× | 1.00× |

The agent beat the author's best-known strategy on every headroom task, so "certified headroom" is
a floor, not a ceiling — see [COMPILER_GAPS.md](docs/COMPILER_GAPS.md) for what to measure instead.
Lazy wrappers (`torch.compile` modes, CUDA graphs, `multi_dot`) never exceed 1.05× where the agent
gets 7–28×.

## How it fits together

```
taskgen ──▶ task package ──▶ runner ──▶ sandbox (shell + GPU + profilers, no network)
                                          └─ agent writes solution.py
                          evaluator ◀── submission        (fresh trusted container, hidden seeds)
                              └──▶ result.json ──▶ analysis
```

Components share nothing but on-disk contracts; a test fails if one imports another.

## Docs

| File | What |
|---|---|
| [docs/DESIGN.md](docs/DESIGN.md) | Thesis, task formulation, generator, transfer matrices, architecture |
| [docs/COMPILER_GAPS.md](docs/COMPILER_GAPS.md) | Tasks as located gaps in compiler capability; families to build next |
| [docs/EVAL.md](docs/EVAL.md) | Correctness, timing protocol, anti-cheat |
| [docs/ECOSYSTEM.md](docs/ECOSYSTEM.md) | Survey of related work, verified by reading the code |
| [docs/QUANT_STUDY.md](docs/QUANT_STUDY.md) | Quantization study design, model + arms (parked) |
| [docs/PLAN.md](docs/PLAN.md) | Milestones, gates, kill criteria, risks |
| [docs/DECISIONS.md](docs/DECISIONS.md) | Append-only decision log (D-1 … D-33) |
| [docs/BUDGET.md](docs/BUDGET.md) | Cost breakdown and levers |
| [docs/ACCESS.md](docs/ACCESS.md) | Accounts, infra state, spend |
| [docs/LOG.md](docs/LOG.md) | Lab notebook: every run, number, surprise and dead end |

## Running it

```bash
uv sync && uv run pytest              # everything but the GPU tests
uv run hotloop-taskgen list           # families and regimes
uv run hotloop-vm launch gpu_1x_a100_sxm4 --name bench     # rent a box (Lambda)
uv run hotloop-run prepare --host <ip>                     # build image, lock clocks
uv run python scripts/run_trajectory.py --host <ip> --task benchmark/dev-v0/<id> \
    --model openai/gpt-5.5 --out runs/
```

Needs `LAMBDA_API_KEY` and `OPENAI_API_KEY` in `.env` (see `.env.example`). Reference strategies,
certificates and agent trajectories are deliberately not committed — dev-v0 is an open development
set, so contamination-free evaluation means regenerating tasks from unpublished seeds.
