# Lab notebook

Newest first. Numbers always with conditions (GPU, driver, n, CI). Dead ends belong here too.

## 2026-09-21 (04:10Z) — First M3 batch killed by OpenAI quota; 4 clean trajectories; safeguards added

- Batch of 36 started 02:21Z on 4 A100s. OpenAI began returning `insufficient_quota` intermittently
  within ~25 min and permanently by ~03:06Z. **Only 4 trajectories ended by the agent's own choice
  (`submitted`)**; 5 were cut off mid-work, 27 never got a model response. All 32 quarantined in
  `runs/m3/_failed_insufficient_quota/` (not committed). Details and fixes: D-29.
- The 4 clean ones (3 official evaluations each, all `ok`): bchain vector-on-the-right **22.4×**
  (certified best 18.1×), bchain shared-wide-middle **7.0×** (6.6×), chain lowrank-sandwich **28.1×**
  (23.4×; diagnostic tier), chain as-written-is-optimal control **1.00×** after 50 turns and $2.49
  of trying. With the pilot (11.6× vs 10.9×) that is 4/4 headroom-or-diagnostic tasks where the
  agent *beat my best-known strategy* — early sign that G2(a) "unsaturated" may **fail** for
  gpt-5.5 on dev-v0; wait for the full batch before concluding.
- I mis-reported a truncated run as "agent's final answer was incorrect". It wasn't final. Retracted.
- Relaunched 04:00Z with the self-terminating batch runner: asked for 4 instances, Lambda had
  capacity for 3. 32 trajectories, ~4.5 h expected, 8 h hard deadline.

## 2026-09-21 (02:10Z) — First agent trajectory (cost pilot): gpt-5.5 beats my best-known; my bug cost it ten turns

`openai/gpt-5.5`, reasoning effort medium, Responses API, caps 50 turns / 150k generated tokens, task
`ragged-pool-zipf-heavy-tail-s0` (certified best-known 10.93×, lazy battery 1.00×). A100-40GB sandbox,
no network. Run dir `runs/pilot/` (not committed).

- **Official score 11.57× / 11.59× / 11.60×** (3 fresh-container evaluations, all `ok`, no flags;
  8 pairs each to ±1%). The agent wrote a CUDA C++ extension (JIT-built with `cpp_extension.load`)
  and iterated through ~10 kernel variants, self-checking after each: 6.2 → 8.0 → 8.2 → 9.4 →
  10.8×. **It beat my jagged-Triton strategy** → best-known for this task is now 11.6×. The
  benchmark is not saturated by its author, which is how it should be.
- Behaviour worth noting: turn 2 was measuring the segment-length distribution over several seeds
  before writing any code — regime diagnosis, unprompted.
- Cost: 40 turns, 1.17M input tokens (93% cached), 34k generated, 29 min wall, of which only
  ~6 min was model latency; most of the rest was ~25 self-checks at ~60 s each (every
  `hotloop-eval` re-measures four baselines). ≈ **$2** at litellm's listed gpt-5.5 prices
  (unverified against OpenAI's current price list). Full 12×3 batch ≈ $70–150 API.
- **My bug, found by the agent:** a repo-wide `ruff format` rewrote quote style in the six ragged
  `workload.py` files after sealing and I committed it, so the in-sandbox evaluator refused the
  package ("does not match its content_hash"). The agent spent turns 3–12 diagnosing it, read the
  evaluator source, and worked around it by re-sealing a copy. All three official evaluations
  errored for the same reason. Fix: restored sealed files, `benchmark/` excluded from ruff,
  `tests/test_benchmark_sealed.py` checks every committed package. Pilot re-scored officially
  after re-syncing the correct task files. This trajectory is tainted for turn-efficiency
  analysis; the task is re-run in the batch.
- Improvement noted, not done: cache baseline timings inside the agent's sandbox so a self-check
  costs seconds, not a minute. It would roughly halve GPU time per trajectory.
- Launched 3 more A100s; batch = 12 tasks × 3 reps across 4 hosts, full task budget (100 turns,
  400k generated tokens).

## 2026-09-21 (01:20Z) — M2 done: dev-v0 emitted and certified with the real evaluator; M3 plumbing live

Conditions: third Lambda A100-SXM4-40GB box, torch 2.11.0+cu128, clocks locked.
`hotloop-taskgen emit-all` (12 packages, fp64-calibrated tolerances, 40 s) then `hotloop-certify`
(100 evaluations through `hotloop-eval` at ±2% / ≤30 pairs, ~25 min). **Every one of the 100
evaluations was `ok`** — calibrated tolerances accepted every legitimate reordering/kernel.
Full certificates: `private/cert/m2-dev-v0/`.

| task | kind | headroom (best known) | lazy battery best | note |
|---|---|---|---|---|
| bchain vector-on-the-right | headroom | **18.1×** | 1.01 | no library one-liner applies |
| bchain shared-wide-middle | headroom | **6.6×** | 1.01 | |
| ragged zipf-heavy-tail | headroom | **10.9×** | 1.00 | |
| ragged small-outliers | headroom | **7.5×** | 1.00 | |
| ragged tiny-rare-long-outliers | headroom | **6.1×** | 1.00 | |
| ragged tiny-outliers | headroom | **3.5×** | 1.00 | |
| chain lowrank-sandwich | diagnostic | 23.4× | **22.9 (multi_dot)** | lazy-solvable, as predicted (D-24) |
| chain tall-skinny | diagnostic | 4.4× | **4.4 (multi_dot)** | |
| bchain vector-on-the-left | control | 1.01 | 0.73 | torch.compile is *slower* than eager here |
| chain as-written-is-optimal | control | 1.00 | 1.00 | wrong order → 0.04× |
| ragged uniform-len64 | control | 0.81 | 1.01 | best hand strategy loses |
| ragged uniform-tiny-len4 | control | 0.30 | 1.05 | jagged kernel → 0.03–0.18× |

Regret (best strategy of task A run on task B, relative to B's best): bchain 1.2–22.7×;
chain 1.0–23.5×; ragged headroom tasks 1.28–3.62×, and up to 7.2× onto controls.
Evaluator-grade numbers agree with the exploration harness to within a few percent → **G1 stands
on evaluator-grade data** for 6 headroom tasks in 2 families.

M3 plumbing, all verified on the box before spending API money:
- Benchmark image builds in <3 min; has torch/triton/nvcc/nsys; evaluator installed.
- Agent sandbox: GPU visible, task mount read-only (`Read-only file system`), no network (DNS
  fails), in-container `hotloop-eval --quick` gives 18.5× for the known-best bchain order.
- Fresh-container scoring: 18.69×, 18.91×. Bug found: `--cap-drop ALL` root can't write a results
  dir owned by uid 1000 → world-writable results dir.
- Live OpenAI smoke test (`gpt-5.5`, Responses API): 2 turns, tool calls parsed, usage logged.
- Noticed and recorded: public docs already name dev-v0's winning strategies → D-28.

## 2026-09-21 (00:00Z) — M2 exploration on a second A100: G1 passes

Conditions: Lambda `gpu_1x_a100_sxm4`, different physical GPU (UUID …ab7541b6) from the M1 box, same
CPU/driver, torch 2.11.0+cu128, triton 3.6.0, clocks locked, TF32 off. `scripts/explore_regimes.py`
(exploration timing: fixed inputs, blocked medians — **not** the evaluator). Data: `docs/data/m2/`.
Cost $0.56 (16.8 min). Total project spend $1.37.

**Cross-box G0 retest** (evaluator GPU suite ×3): all green. Fused-Triton toy score 1.053, 1.057,
1.114 here vs 1.045–1.067 on box 1 → 7 runs, mean ≈ 1.065, SD ≈ 2.3%, one ~5% outlier. Inside G0's
3% but single-run CIs (±0.9%) clearly understate it → D-22 (pool ≥3 fresh-process runs) is
necessary, not optional. The toy is a 55 µs host-launch-bound call, i.e. the noisiest kind.

**Matrix chain** (fp32, ms/call; best automatic = eager everywhere except noted):

| regime (dims) | best auto | best order | headroom | multi_dot |
|---|---|---|---|---|
| 8192·64·8192·64·16 | 1.034 | `(0(1(23)))` 0.036 | **28.9×** | 0.037 |
| 4096·4096·32·4096·4096 | 7.428 | `((01)(23))` 0.266 | **28.4×** | 0.262 |
| 8192·128·4096·128·2048 | 1.246 | `(0((12)3))` 0.272 | **4.6×** | 0.272 |
| 2048·8192·8192·2048·1 | 18.34 | `(0(1(23)))` 0.300 | **61×** | 0.300 |
| 16·8192·64·8192·8192 | 0.272 (max-autotune) | as written 0.298 | 0.91× (control) | 0.298 |
| 512⁵ / 64·4096³·64 | 0.082 / 0.301 | as written | 1.00× (control) | = |

Regret of using one regime's best order in another: up to **197×**. torch.compile never
re-associates. **But `multi_dot` ≈ optimal everywhere → the pure family is lazy-solvable** (D-24).

**Ragged softmax-pooling** (fp16, packed `(v, s, offsets)`, padded reference; ms/call):

| regime | pad waste | best auto (max-autotune) | winner | headroom | runner-up |
|---|---|---|---|---|---|
| tiny(4)+outliers(1024), S=262k | 225× | 4.497 | scatter ops, compiled 0.380 | **11.8×** | jagged b16 0.524 |
| tiny(4)+outliers(256), S=262k | 60× | 1.537 | scatter ops, compiled 0.436 | **3.5×** | jagged b16 0.603 |
| tiny(2)+outliers(128), d=8 | 60× | 0.746 | scatter ops, compiled 0.247 | **3.0×** | scatter eager 0.526 |
| small(16)+outliers(1024) | 57× | 2.050 | jagged Triton b16 0.256 | **8.0×** | jagged b128 0.341 |
| zipf heavy tail | 84× | 2.003 | jagged Triton b128 0.180 | **11.1×** | jagged b1024 0.254 |
| zipf heavy tail, d=256 | 48× | 1.253 | jagged Triton b128 0.234 | **5.4×** | scatter compiled 0.492 |
| bimodal 1% long | 64× | 1.001 | jagged b128 0.141 | **7.1×** | jagged b1024 0.142 |
| uniform len 64 (control) | 1× | 0.070 | — (jagged 0.094) | 0.75× | |
| many tiny len 4 (control) | 1× | 0.088 | — (scatter 0.343; jagged b128 1.232 = **0.07×**) | 0.26× | |

- Two-sided flip found on the second try. First grid had only one winner among headroom regimes;
  the data hinted scatter beats jagged for tiny segments, but I'd only tested that with zero
  padding waste (compiler wins). Adding rare long outliers gave waste *and* tiny segments → flip.
- Block size alone is a regime effect: jagged b1024 is 54× slower than b128 at d=256.
- Negative result: length-bucketed dense never wins anywhere (host sync + many small launches).
- `max-autotune-no-cudagraphs` is the best automatic baseline at every ragged point and is
  8–30× faster than eager on padded regimes — D-4 (score against the compiler) is doing real work.

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
