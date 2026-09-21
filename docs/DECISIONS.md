# Decision log

Append-only. Supersede, don't rewrite. Status: `accepted` | `provisional` (expected to be revisited
with data) | `superseded by D-n`.

---

## D-1 · Core claim is "regime-flip tasks", not "more kernel tasks" — accepted · 2026-09-20
**Context.** Ecosystem already has fixed-task kernel benchmarks, multi-shape eval, roofline scoring,
anti-cheat (ECOSYSTEM.md). None generate tasks; none make the workload change the *right answer*.
**Decision.** The research contribution is a generator + certification procedure for tasks whose
optimal strategy flips across workload parameters, and transfer-matrix scoring built on it.
**Why.** It's the only formulation here that (a) is novel, (b) is falsifiable cheaply before any
agent runs, (c) yields a behavioural measure of feedback use that the quant study needs.
**Kill criterion.** If certified cross-regime penalties are < 1.25× for most families on real
hardware (gate G1), the premise is wrong → pivot to hardware-flip or composition-only tasks.

## D-2 · Free-form agent, single deliverable — accepted · 2026-09-20
Agent gets shell + GPU + profilers + `hotloop check`; deliverable is a dir with
`solution.py: run(*inputs)`. Any backend. No prescribed tools/workflow (contrast TritonGym's fixed
tool API, Apex's MCP scaffolding). **Why:** we measure engineering, not tool-following; keeps the
benchmark agnostic to harnesses, including future ones.

## D-3 · Workload = weighted set of input generators incl. data distribution — accepted · 2026-09-20
Not a single shape. Data distribution (raggedness, skew, sparsity, routing) is a first-class
parameter because it drives performance for the most interesting families. Hidden seeds at final
eval; shapes/distributions public. **Why:** matches real perf work ("here's the trace"), defeats
memoisation, and lets specialisation/dispatch be a legitimate strategy.
**Open:** should some tasks hold out *shapes* too (AgentKernelArena found agents hardcode shape
assumptions)? Provisional: no for regime tasks (the point is specialising to a stated regime);
yes as a separate generalisation track later.

## D-4 · Baseline = strongest automatic baseline, same session — accepted · 2026-09-20
`min(eager, torch.compile default, max-autotune ± cudagraphs)` plus the "lazy battery"
(DESIGN §3.3.5), measured interleaved with the candidate. Secondary: roofline fraction
(Atrex/SOL-ExecBench style). **Why:** eager baselines inflate results (KernelBench authors now say
so themselves); same-session ratios cancel box-to-box drift.

## D-5 · Two-level generator: certified motif families, then composition grammar — accepted · 2026-09-20
Level 1 has hand-written reference strategies (hidden) → strong certificates (headroom + regime
flip). Level 2 composes motifs via a typed DAG grammar → admitted via automatic headroom
certificate only. **Why:** pure random DAGs (cf. DRTriton) give volume without performance
structure; pure hand-written families aren't "procedural". Level 1 validates the premise; Level 2
scales it. MVP is Level 1 only.

## D-6 · Budgets in turns/tokens + sandbox GPU-seconds, not wall-clock — provisional · 2026-09-20
**Why:** wall-clock couples score to serving throughput, which would confound the quant study
(quantized models decode faster). Wall-clock cap exists only as a safety limit.
Initial numbers to pilot: 100 turns, 1.5M total tokens, 30 GPU-min.

## D-7 · Reuse map — accepted · 2026-09-20
See ECOSYSTEM.md "Reuse decisions". Headlines: Atrex-style task dir; timing = GPU MODE `eval.py`
*design* (licence forbids code reuse) + ABBA interleave; anti-cheat = union of published exploit
catalogues as a regression suite; tolerance calibration à la SOL-ExecBench; no Redis/FastAPI
backend for MVP; mini-swe-agent-style harness for controlled runs.

## D-8 · Infra: Modal (sandboxes + serving) + one root-VM provider (reference timing) — superseded by D-13 · 2026-09-20
**Why Modal:** best programmatic parallel sandboxes, per-second billing, used by GPU MODE whose
dataset card calls Modal timings more trustworthy than their on-prem box. One account covers
serving too. **Why a root VM as well:** can't lock clocks or guarantee perf counters under gVisor,
and gVisor may tax host-side launch overhead — which is one of our regime axes. Probes in
ACCESS.md decide whether scores of record can come from Modal. Cost-optimise serving to RunPod
later only if spend matters.

## D-9 · Quant study model: Qwen3.8-27B dense (fallback Qwen3.6-27B) — provisional · 2026-09-20
**Why:** BF16-native (honest baseline), dense (no routing confound), BF16 fits one H100 so every
arm runs TP=1 on identical hardware, reputable FP8/GPTQ/AWQ/NVFP4/GGUF checkpoints exist, and
reported agentic-coding strength suggests it clears the floor. Natively low-precision models
(gpt-oss, Devstral-2, Kimi) are excluded because they have no true higher-precision reference.
Phase-1 arms: BF16×2 (null), FP8, W4A16-GPTQ, W4A16-AWQ. Sub-4-bit deferred (needs llama.cpp +
bridge arms). **Revisit if:** G2 pilot shows BF16 scores at floor on our tasks (then tasks need an
easier tier, or the model must be bigger), or vLLM #55766 blocks prefix caching on 3.8.

## D-12 · A flat headline score is not a null result — accepted · 2026-09-20
Following 2607.27275, the quant study reports per-channel failure rates (format, hallucinated
API, perseveration, abandon-known-good, regime regret) and tokens/turn alongside score.
Raw completions are logged and parsed by us, not only by the serving engine's tool parser.

## D-10 · Score is 0 on any correctness failure; final submission is what counts — accepted · 2026-09-20
No credit for best-intermediate. **Why:** shipping a regression as the final answer is a real
engineering failure and plausibly precision-sensitive (hygiene metric). Best-seen is still
recorded for analysis.

## D-11 · Forward-only, single-GPU, inference-style tasks for MVP — accepted · 2026-09-20
Backward passes, multi-GPU, and memory-capped variants are later axes. **Why:** smallest version
that can falsify D-1.

## D-13 · Infra: root VMs for everything (Lambda first), provider-agnostic runner — provisional · 2026-09-20
**Supersedes D-8.** Prompted by Emaan asking why not use Lambda for everything.
**Decision.** Agent sandboxes, final evaluation, and vLLM serving all run on root GPU VMs
(Lambda first; Crusoe/Hyperstack as alternates). Agent runs in Docker (`--gpus`) on the VM; final
eval in a fresh container on the same SKU; serving on a *separate* VM. Provisioning/teardown/
autostop via SkyPilot rather than our own code. The runner's only infra assumption is "a Docker
host with an NVIDIA GPU reachable over SSH", so Modal or any other provider can be added as a
burst backend without touching benchmark code.
**Why.** (1) The environment the agent profiles in must be the environment it is scored in —
otherwise launch-bound decisions are made against different host overheads (gVisor) than the
scoring box. (2) Root gives locked clocks, guaranteed CUPTI counters, pinned driver. (3) Cost is
about equal. (4) Scale is small (~300–400 GPU-h of trajectories; 4–8 VMs).
**Costs accepted.** We own VM lifecycle (leak risk → SkyPilot autostop + on-box idle watchdog +
spend log); Lambda capacity is flaky (→ provider-agnostic runner); no L40S on Lambda (hardware
axis becomes A100 vs H100 / A10 / GH200).
**Unverified, probe day 1:** Lambda VMs permit `nvidia-smi -lgc/-pm` and
`NVreg_RestrictProfilingToAdminUsers=0`. If not → Crusoe (documents clock locking).

## D-14 · Primary bench SKU = A100-40GB SXM4 on Lambda; H100 is the second SKU — provisional · 2026-09-20
**Context.** Live Lambda API (2026-09-20): `gpu_1x_a100_sxm4` $1.99/h, available in 3 regions.
1× H100 (PCIe $3.29, SXM5 $4.29) **no capacity anywhere**; only 2× H100 ($8.38/h) available.
8× nodes carry **no per-GPU discount** (8×H100 $3.99/GPU, 8×A100 $1.99/GPU).
**Decision.** Certify tasks and run agent trajectories on A100-40GB. H100 becomes the
hardware-axis second SKU, run opportunistically when 1× capacity appears.
**Why.** Availability is the binding constraint, and it's half the price. Scientifically fine:
tasks are bf16/fp16/fp32 (no FP8 needed on the bench GPU), Triton/torch.compile are mature on
Ampere, 40 GB just bounds workload sizes (generator already checks memory fit).
**Consequence.** Serving still needs ≥ 80 GB Hopper for BF16-27B and FP8 arms → 2× H100 at
$8.38/h today, or GH200 ($2.29/h, 96 GB, aarch64 — vLLM support to verify) / 1× H100 when
available. Revisit serving provider at M3.

## D-15 · Frontier reference agent may be any vendor (OpenAI key offered) — accepted · 2026-09-20
Role is ceiling + discriminative check + red-teaming the evaluator (PLAN M3); vendor-neutral.
Uses the same OpenAI-compatible harness path as the open-weight model.

## D-16 · Public repo; answer-key material kept out of it — accepted · 2026-09-20
**Context.** Emaan had already created `emaanh/HotLoop` as **public** (and set it as `origin`); I
had proposed private without checking. Asked why private.
**Decision.** Repo stays public: docs, taskgen, evaluator, runner, agents. Kept out (gitignored
`private/`, later a separate private repo): hidden **reference strategies**, **hidden eval seeds**
of any frozen task set, and certification data that reveals best-known solutions.
**Why.** Public is the right default for research. The only real risk is contamination/lookup of
the answer key. Procedural generation is itself the defence: the generator is public, a frozen
set's seed is not, and fresh tasks can be regenerated from an unpublished seed.
**Consequence for the runner.** Agent sandboxes get **restricted network egress** (package
indexes only, no general web/GitHub), otherwise an agent could fetch this repo or published
kernels mid-run. Log all egress attempts as a trajectory signal.

## D-17 · Current scope: through "OpenAI model evaluated on the benchmark", then stop — accepted · 2026-09-20
**Directive from Emaan.** Build M0 → M1 → M2 → M3 with the **OpenAI frontier model only**, then
stop and report. Out of scope until told otherwise: open-weight models, vLLM serving, M4
(generator scale-up), M5 (quantization study).
**M3-lite** = runner + minimal harness + OpenAI agent + scripted "lazy" agent on the ~12 certified
tasks × 3 seeds. Gate G2 is evaluated on what this can show: unsaturated, lazy < frontier,
regime regret observed, zero reward hacks on audit, plus run-to-run variance for later power
analysis. (The "open model not at floor" check is deferred with M5.)
**Budget under this scope:** GPU ≈ $250 (A100-40GB @ $1.99/h) within tranche T1 = $300;
OpenAI API ≈ $350 (unverified guess — measure on first 3 trajectories and re-estimate before the
full batch). No HF token / serving VM needed.

## D-18 · Agent harness: depend on mini-swe-agent 2.4.6, don't write our own loop — accepted · 2026-09-20
**Context.** M0 code reading: ~500-line MIT core, protocol-based, injectable classes; native bash
tool calls; Responses-API model class for OpenAI reasoning models; tenacity retries; trajectory
keeps the full raw response + usage per step (what D-12 needs).
**Decision.** Pin `mini-swe-agent==2.4.6`. We write: `SshDockerEnvironment`, an agent subclass
(token + GPU-second budgets, per-command durations), a YAML config (head+tail truncation), and a
converter to our `trajectory.jsonl`.
**Known defects to engineer around.** Docker env timeout kills the local `docker exec` client,
not the in-container process → wrap commands in `timeout -k 5 N bash -lc` inside the container.
Stateless shell per command (the prompt must say so). litellm is heavy and has had bad releases →
exact pins, lockfile.
**Alternative rejected.** Own ~200-line loop over the OpenAI SDK: full control, but we'd re-solve
Responses-API plumbing, retries and format-error handling for no research value.

## D-19 · Evaluator timing = host wall clock + device-wide sync, block-interleaved, fresh values every call — accepted · 2026-09-20
**Context.** M0 code reading of how others time: CUDA events on the current stream (KernelBench,
GPU MODE; exploitable via side streams — KernelBench's own code has a TODO admitting it), CUPTI
kernel spans (SOL-ExecBench; all streams but excludes host time), `do_bench` (Atrex default).
Interleaving, where it exists (Atrex), is process-level ABBA.
**Decision.** Primary metric: `perf_counter` around `run(*inputs)` + `torch.cuda.synchronize()`
(device-wide → side-stream work is always counted; host overhead is counted, which launch-bound
regimes need). Baseline and candidate alternate in **blocks within one session** (finer than
Atrex's process-level ABBA) with the paired bootstrap estimator in `stats.py`. Every timed call
gets **fresh values at a fresh address** from a pre-generated pool keyed by an HMAC of a secret
seed; outputs of timed calls are spot-checked against the reference afterwards. CUDA-event and
CUPTI timings are recorded as *diagnostics*; a large disagreement with wall time raises a flag.
**Why not CUPTI as primary.** It would make launch-bound tasks unscoreable and pins us to one
CUDA/cupti-python stack.

## D-20 · No SkyPilot; own thin Lambda client; API key never touches a VM — accepted · 2026-09-20
**Amends D-13.** SkyPilot's autodown works by copying provider credentials onto the VM so the
VM can terminate itself. Bench VMs host untrusted agents with a shell → the Lambda key could be
exfiltrated. **Decision:** `hotloop-vm` (stdlib, ~150 lines) drives the Lambda API from the
operator machine only. **Cost accepted:** no on-box auto-terminate. Mitigations: every launch is
written to a local ledger (`~/.hotloop/vm_ledger.jsonl`), `hotloop-vm reap --older-than-hours H`,
instances are terminated at the end of each work session, and live instances are listed in
ACCESS.md. Emaan can always check/kill at cloud.lambda.ai/instances.

## D-21 · Timed region = block + late-revealed stash into driver-owned poisoned buffers, minus trusted overhead — accepted · 2026-09-20
**Refines D-19** after running the evaluator on a real A100.
**Hole found (by inspection, before it was exploited).** The submission shares a process with
the worker and can rebind `torch.cuda.synchronize` to a no-op, so "done" arrives while kernels
are queued. **Fix:** the clock stops only after outputs `[last, random j]` (indices sent *after*
the block ran) are copied on the worker's current stream into **driver-owned, NaN-poisoned
buffers shared once per entry**, and the driver has cloned those buffers and synchronised its own
context. Pending work is either waited for or observed as poison → incorrect. The worker also
calls captured-original sync functions and reports rebinding (fatal `sync_tampered` flag), but
the score does not depend on that.
**Measured on A100 (16 × 65 µs calls):** empty RPC 72 µs; fetching 2 outputs via fresh CUDA-IPC
handles **341 µs**; clone+sync 55 µs → 35% overhead on a 1 ms block. Hence: no IPC handle
creation in the timed path; input pool sized from device memory (25%) so blocks are long even
when inputs are large and never reused; fixed round-trip cost measured on the **reference**
worker only (a submission can't inflate it) and subtracted from both sides.
**Result:** CI half-width 4.5% → 0.9% in 7–11 s; spurious `clock_disagreement` flags gone.

## D-22 · One evaluation's CI is not the whole uncertainty: pool fresh-process evaluations — provisional · 2026-09-20
Same solution, same box, 4 separate evaluations: 1.048, 1.067, 1.045, 1.061 (each ±0.9% CI).
Between-process SD ≈ 1% is not visible inside a session (codegen/allocation/CPU placement differ
per process). **Decision:** scores of record = geometric mean over ≥3 evaluations in fresh
processes, reported with the between-run spread; minimum claimable effect 3%. **To investigate:**
core pinning for workers; whether the variance comes from the compiled baseline or the candidate.

## D-23 · Gate G1 passes on exploration data; D-1 stands — accepted (provisional on evaluator-grade re-measurement) · 2026-09-21
**Evidence** (`docs/data/m2/`, A100-40GB, torch 2.11+cu128, exploration timing, fixed inputs):
- *Matrix chain* `A@B@C@D`: eager = all three torch.compile modes at every point (the compiler
  never re-associates). 4 regimes with headroom 4.6× / 28× / 29× / 61× over the best automatic
  baseline, 4 different argmax orders, controls at 1.00×, cross-regime regret up to 197×.
- *Ragged softmax-pooling* (packed inputs, padded reference): 3 regimes with ≥3× headroom over
  `max-autotune` and **three different winners** — tiny segments + rare outliers → compiled
  scatter/segment ops (3.0–11.8×); small segments + outliers → jagged Triton, BLOCK=16 (8.0×);
  heavy-tailed → jagged Triton, BLOCK=128 (11.1×). Cross-regime regret 1.3–6.8×. Where padding is
  free (uniform lengths) the compiler wins and the jagged kernel scores 0.75× to 0.07×.
  Length-bucketed dense never wins.
**Verdict.** G1 (a) argmax changes, (b) penalty ≥1.25×, (c) ≥1.3× headroom in ≥2 regimes: met by
2 of 2 families explored (gate asked for ≥2 of 3). The premise of D-1 is not falsified.
**Caveats that stay attached to this claim.** Exploration harness, not the evaluator; one box;
flips are relative to *my* strategies — a better kernel could dominate several regimes and erase
a flip, so regret must be recomputed as best-known moves.

## D-24 · Family decisions after G1 — accepted · 2026-09-21
1. **Pure matrix chain is lazy-solvable**: `torch.linalg.multi_dot` is within 3% of the best
   order everywhere. It fails anti-triviality (DESIGN §3.3.5). → `multi_dot` joins the lazy
   battery; the family becomes **algebraic structure** with the plain chain as a *diagnostic
   tier* ("did the agent look at shapes at all?") and disguised members where no library
   one-liner applies (`diag(A@B@C)`, `((A@B)*M).sum()`, low-rank-plus-identity powers, Kronecker
   products, chains broken by cheap commuting ops).
2. **Ragged pooling** is the first full family: regimes = {tiny+outliers, small+outliers,
   heavy-tail} as sibling tasks, plus {uniform} as a **no-headroom control**.
3. **No-headroom controls are kept in the task set** (not rejected by the headroom certificate as
   DESIGN §3.3 originally said): max score ≈ 1.0, but they are where pattern-matching agents
   visibly lose (0.07–0.75×). They are scored and reported separately from headroom tasks.
4. Third family still to find; candidates: multi-tensor update (horizontal fusion), pairwise
   top-k. Not needed to proceed to M3-lite.
