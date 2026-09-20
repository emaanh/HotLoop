# Evaluation: correctness, timing, anti-cheat

Status: v0 draft (2026-09-20). To be revised after ecosystem survey (reuse timing utilities where
they are already good — see ECOSYSTEM.md).

## Contract

`hotloop-eval --task <task_pkg> --submission <dir> --out result.json`

Runs in a **fresh sandbox the agent never touched**, same SKU + image as the agent's sandbox.
Only the submission directory crosses over. Pre-built binaries are allowed (agent may have
compiled CUDA extensions) but a `build.sh` is preferred and, if present, is re-run; build time is
budgeted, not scored.

## Correctness

- Oracle = `reference.py` run in a **separate process** from the candidate (the Sakana CUDA
  Engineer incident: a candidate read the reference's result out of reused allocator memory).
  Candidate runs first; allocator is poisoned (fill freed blocks with NaN) between runs.
- Inputs from **hidden seeds**, regenerated per trial. Shapes/distributions follow the public
  workload spec; values never repeat → output caching and constant-folding of data are useless.
- Input classes per workload entry: typical draws; boundary draws of the declared distribution
  (e.g. max-length sequence, empty segment, all-same index); numerically hostile draws (large
  magnitudes, to catch unstable softmax/variance rewrites) when the reference itself is stable
  on them.
- Tolerance: per-task, derived at certification time. Reference is re-run in fp64; tolerance =
  max(dtype default for `assert_close`, *k* × observed |ref_native − ref_fp64|). This admits
  legitimate re-association/fused-precision differences and rejects "cast everything to fp16".
- Also checked: output shape/dtype/device/stride contract, inputs not mutated (unless declared),
  no NaN where the reference has none, determinism across two runs if the task declares it.
- Outputs from *timed* iterations are spot-checked too (a solution can't be correct in the check
  phase and sloppy in the timed phase).

## Timing

- Exclusive whole GPU, no MIG/MPS/time-slicing; one evaluation per box at a time.
- Record and store with every result: GPU SKU, VBIOS, driver, CUDA, torch/triton versions, CPU
  model, clocks & throttle reasons before/after, temperature, ECC state, persistence mode.
  Lock clocks (`nvidia-smi -lgc`) where the provider permits.
- Metric is **end-to-end wall time per call with a trailing synchronize**, measured by CUDA
  events *and* host timers. Host overhead is part of the problem (launch-bound regimes), so CPU
  model is part of the hardware pin.
- Protocol: warmup until steady (compile/autotune excluded, capped), then *interleaved* ABAB
  blocks of baseline and candidate within the same process group/session so drift cancels in the
  ratio. Fresh inputs per block. Report median, IQR, bootstrap 95% CI of the **ratio**.
- A result is `inconclusive` (re-queued) if CI width exceeds threshold or throttling is detected.
- Baselines are re-measured in every evaluation session, never cached across boxes.
- Peak memory is recorded; tasks may declare a memory cap (e.g. ≤ reference peak) so "trade
  unlimited memory for speed" is an explicit, per-task choice.

## Anti-cheat battery (evaluator's own test suite)

Adversarial "null agents" that must all score 0 / be flagged. Shipped as regression tests:

1. Output memoisation keyed on input hash / pointer / shape.
2. Returning uninitialised memory hoping to alias the reference result.
3. Monkeypatching `torch.cuda.Event`, `time`, `torch.cuda.synchronize`, or the evaluator module.
4. Async trickery: returning before work completes; doing work on a side stream without sync.
5. Detecting the evaluator (env vars, call stack, iteration count) and switching behaviour.
6. Precision dumping beyond tolerance; skipping work on "typical" values.
7. Warmup-phase precomputation that depends on leaked seeds.
8. Forking background processes that pre-compute on the GPU.

Mitigations: candidate in subprocess with evaluator-owned timers outside it (host wall clock
around an RPC + device sync performed by the evaluator's own CUDA context check), hidden seeds
delivered per-call, randomised iteration counts, import-time integrity check of torch timing
symbols, process-tree and GPU-process audit, timed-iteration output verification.

**Principle:** the agent's sandbox is *untrusted*; the evaluator sandbox is trusted and minimal.

**Network:** agent sandboxes have restricted egress (package indexes only). The repo is public
(D-16), so open internet would let an agent fetch reference material mid-run. Egress attempts are
logged in the trajectory.

## What the agent sees

`hotloop check` inside the agent sandbox = same code path, public seeds, reports correctness and
speedup vs baseline with CI. Agents are told explicitly that final scoring uses hidden seeds on a
fresh machine of the same SKU. Profilers available: `torch.profiler`, `nsys`, Triton's
proton/autotune logs, `compute-sanitizer`; hardware counters (CUPTI) if the provider allows.
