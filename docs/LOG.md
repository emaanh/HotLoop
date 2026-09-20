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
