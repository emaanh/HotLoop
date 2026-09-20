# Lab notebook

Newest first. Numbers always with conditions (GPU, driver, n, CI). Dead ends belong here too.

## 2026-09-20 (evening) — M1 on a real A100: G0 passes within-box

Conditions: Lambda `gpu_1x_a100_sxm4` us-west-2, A100-SXM4-40GB, driver 570.148.08, KVM, EPYC 7J13
30 vCPU. Raw data in `docs/data/m1/`. Cost $0.82 (24.6 min).

**Raw timing noise** (`scripts/probe_gpu.py`, host torch 2.7.0, 45 s per class, ~2.3–2.8k blocks):

| class | block CV unlocked → locked | A-vs-A ratio (all / first 30 pairs) | planted 1.10× (first 30 pairs) |
|---|---|---|---|
| compute (10× 4096² fp16 matmul) | 1.58% → 0.86% | 1.0000 ±0.00% / ±0.03% | 1.0995 [1.0993, 1.0996] |
| bandwidth (10× pointwise on 64M fp16) | 0.09% → 0.08% | 1.0000 ±0.00% / ±0.05% | 1.0994 [1.0993, 1.1002] |
| launch (1000 tiny kernels) | 0.51% → 0.47% | 1.0001 ±0.02% / ±0.17% | 1.1009 [1.0991, 1.1022] |

- ~1% thermal warm-up drift on compute (31→53 °C) is visible in raw times and **cancels in the
  paired ratio**, as designed. Clock locking halves raw compute CV; paired ratios barely need it.
- **G0 thresholds (CV ≤ 3% device-bound, ≤ 8% launch-bound; planted 1.10× detected) are met with
  >10× margin within one box/session.** Not yet tested: across fresh VMs (needs a second box).

**Evaluator on GPU** (torch 2.11.0+cu128, triton 3.6.0, clocks locked), toy task
`sum(relu(x*a+b)^2,-1)` on 4096², 3 consecutive suite runs, all 6 tests green each time:

| submission | verdict | speedup vs best baseline (`compile_default`) |
|---|---|---|
| fused Triton kernel | ok | 1.067 / 1.045 / 1.061 (each ±0.9%); **16× vs eager** |
| same, work on a side stream (CUDA-L1 exploit) | ok, no gain | 0.84 / 0.86 / 0.88 |
| rebinds `torch.cuda.synchronize` to no-op | **flagged**, score 0 | — |
| copy of the eager reference | ok | 0.123–0.128 |
| honest kernel, worker sync disabled entirely | **incorrect** (poison observed) | — |

- The D-4 point in one line: a kernel that is 16× faster than eager is 1.05× vs the compiler.
- First attempt had ±4.5% CIs: fresh-values-per-call capped blocks at 16 calls ≈ 1 ms while
  CUDA-IPC handle creation for fetched outputs cost 341 µs/block. Redesign → D-21 → ±0.9%.
- Found and closed a real hole by inspection (no-op synchronize) → D-21.
- Between-process spread (~1% SD) exceeds the within-run CI → D-22.
- Surprises: PyPI torch now targets CUDA 13 and rejects driver 570 → cu128 index. Lambda's API
  403s urllib's default User-Agent. Boxes ship Python 3.10, no nsys.

## 2026-09-20 — Kickoff, planning only

- Repo empty; local machine has uv/docker/gh, no GPU tooling or API keys (ACCESS.md).
- Three parallel web surveys (ecosystem, quantized serving, cloud GPUs) → ECOSYSTEM.md,
  QUANT_STUDY.md, ACCESS.md. All are secondary-source summaries; M0 verifies what we depend on.
- Surprises worth remembering:
  - The field moved a lot in 2026: Atrex-Bench, SOL-ExecBench, AgentKernelArena, FastKernels
    already do multi-shape eval, roofline scoring and serious anti-cheat. Our novelty must be the
    *generator + regime-flip certification*, not the harness. (D-1)
  - CUDA-L1 reported 32.8% of its KernelBench "wins" used a side-stream timing exploit. SOL-ExecBench
    flagged 14.5% of submissions. Reward hacking is the default outcome, not an edge case.
  - CPU-side perf benchmarks failed replication badly (39/102 GSO tasks valid cross-machine) →
    minimum effect sizes and per-task CIs are non-negotiable (G0).
  - GPU MODE found Modal timings *more* trustworthy than their on-prem node. Cloud ≠ noisy by default.
  - "Flat Score, Amplified Failures": quantized agents can hold headline score while failure
    channels amplify → D-12.
  - GPU MODE's eval code has a restrictive custom licence — design only, no code reuse.
- Decisions D-1…D-12 recorded. No spend.
- Later same day: Emaan asked why not Lambda for everything. Good challenge — the agent should be
  scored in the environment it profiled in, and root VMs are the trustworthy one. **D-13 supersedes
  D-8**: root VMs for all GPU work (Lambda first, SkyPilot lifecycle), Modal dropped as primary.
- Budget redone bottom-up (BUDGET.md): ≈ $4,200, not $2,500 — first figure ignored that bench GPUs
  idle during model turns (~80% of cost). Levers bring it to ≈ $2,200–2,600. Staged approval proposed.
- Lambda key received (moved `key.txt` → `.env`, 0600, gitignored; `key.txt` deleted after verifying
  match). API verified. SSH pubkey uploaded. Live price/capacity snapshot in ACCESS.md:
  **no single-H100 capacity anywhere; A100-40GB $1.99/h available in 3 regions; no multi-GPU
  discount.** → D-14 (A100-40GB primary SKU), BUDGET v2 ≈ $2,950 (G1 for ≈ $150).
- Emaan will provide an OpenAI key for the frontier reference (D-15). Staging "sounds good".
- OpenAI key added and verified (models list OK). Remaining asks: HF token (M3), permission
  allowlist OK, go-ahead to commit + start M0.
- Repo is public and was already `origin` (I hadn't checked) → D-16: answer-key material stays in
  gitignored `private/`; agent sandboxes get restricted egress. Initial docs commit a34d1ad (local).
- **Scope set by Emaan (D-17): go through evaluating the OpenAI model on the benchmark, no further.**
- M0 started: verification reads of reuse targets (subagents) + repo skeleton/schemas.
- **M0 verification done** (11 repos cloned and read, nothing run). ECOSYSTEM.md v1 has the
  corrections table. Headlines: SOL-ExecBench does *not* calibrate tolerances (ours is original);
  nobody combines secret seeds + fresh values per timed call + all-stream timing; FastKernels has
  no agent runtime or nsys; KernelGYM has no LICENSE file; mini-swe-agent is a good dependency
  with one timeout bug. → D-18 (harness), D-19 (timing method).
- Code landed (all CPU-tested, 22 tests): contracts + JSON Schemas + boundary test; `stats.py`
  (paired-block bootstrap ratio; synthetic tests show shared 30% drift cancels, 1.10× detected
  ≥95% at CV 3%/30 pairs and CV 8%/120 pairs); `correctness.py` (contract checks + fp64 tolerance
  calibration: accepts reordered stable softmax on unseen seeds, rejects half-precision dumping).
