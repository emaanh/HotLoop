# Lab notebook

Newest first. Numbers always with conditions (GPU, driver, n, CI). Dead ends belong here too.

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
- Next: access from Emaan → M0 can start immediately without it.
