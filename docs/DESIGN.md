# HotLoop — Design

Status: v0 draft (2026-09-20). Decisions referenced as `D-n` live in [DECISIONS.md](DECISIONS.md).

## 1. One-paragraph thesis

GPU performance engineering is not "write a fast kernel"; it is *diagnosing which regime a
workload lives in and choosing the optimization that regime rewards*. The same PyTorch program
is launch-bound at one shape, bandwidth-bound at another, and wants an algorithmic rewrite at a
third. Existing kernel benchmarks mostly fix one shape per op and score against eager PyTorch, so
they measure kernel *transcription* ability, not engineering judgment. HotLoop's claim:

> We can **procedurally generate** tasks whose optimal optimization strategy **provably flips**
> with workload parameters, verify this mechanically on real hardware, and use the resulting
> benchmark to measure capabilities (diagnosis, recovery, feedback use) that fixed-shape
> benchmarks cannot see.

Everything in the MVP exists to falsify or validate that claim cheaply.

## 2. Task formulation

A **task** is a tuple `(program, workload, hardware, budget)`:

- **program** — a PyTorch function `reference(*inputs) -> outputs`. Defines semantics. It is the
  correctness oracle and is written naturally (the way a modeller would write it), not adversarially
  slow.
- **workload** — a *weighted set of input generators*, not a single shape. Each entry fixes
  shapes, dtypes, strides/layout, and the **data distribution** where it affects performance
  (sequence-length raggedness, index skew, sparsity pattern, routing imbalance). Values are drawn
  from a seed the agent never sees at final evaluation. Think "shape histogram from a serving
  trace". (D-3)
- **hardware** — a pinned GPU SKU + software image. The same (program, workload) on a different
  SKU is a *different task*.
- **budget** — wall-clock, GPU-seconds, and model tokens/turns. (D-6)

The agent gets a sandbox with a real exclusive GPU, a shell, the task directory, compilers,
profilers, and a local `hotloop check` command (correctness + timing on public seeds). It may do
anything. The **only** deliverable is a directory containing `solution.py` exposing
`run(*inputs)`; backends are free (Triton, CUDA/C++ extensions, torch.compile, CUDA graphs,
cuBLASLt calls, algebraic rewrites in plain PyTorch). No prescribed workflow. (D-2)

### Score

Per workload entry *w*: `speedup_w = t_baseline_w / t_candidate_w`, where the baseline is the
**strongest automatic baseline** — `min(eager, torch.compile default, torch.compile
max-autotune[-no-cudagraphs|+cudagraphs])` measured in the same session on the same box. Beating
eager is not an achievement in 2026; beating the compiler is. (D-4)

Task score = weighted geometric mean of `speedup_w`, **0 if any correctness check fails**
(and the solution the agent *last validated* is not substituted — shipping a broken final answer
is a real failure mode we want to measure). We additionally report:

- `headroom_closed` = log(speedup) / log(best_known_speedup), best-known = max over certified
  reference strategies and all solutions ever submitted (GSO-style moving target).
- `regime_regret` — see §4.

## 3. Why synthetic generation, and what kind

Superficial variants (rename ops, jitter shapes ±10%) add rows, not information. The generator's
unit of value is a **regime boundary**: a place in workload space where the ranking of
optimization strategies changes. We build the generator around mechanically locating and
certifying those boundaries.

### 3.1 Regime axes (where "what good looks like" changes)

| Axis | One side | Other side | What flips |
|---|---|---|---|
| Launch vs device bound | many tiny ops/tensors | few large | CUDA graphs / horizontal fusion / host-sync removal vs kernel quality |
| Arithmetic intensity | pointwise/normalization chains | GEMM/conv-like | fusion & bandwidth vs tiling/algorithm/tensor-core use |
| Reduction geometry | huge outer × tiny inner | tiny outer × huge inner | parallelise across rows vs within row; block sizes; multi-pass |
| Materialization | small intermediates | O(N²)/O(N·K) intermediates | naive fine vs streaming/online/recompute (Flash-style) |
| Raggedness | uniform lengths | heavy-tailed lengths | pad+dense vs packed/jagged kernels vs bucketing |
| Index skew | uniform indices | Zipfian / duplicated | scatter/gather contention, sort-based vs hash-based, caching hot rows |
| Structured sparsity | dense mask | block-sparse / banded / causal | dense masked vs skip-block kernels |
| Routing imbalance (MoE-like) | balanced groups | skewed groups | batched-GEMM vs grouped-GEMM vs sort+segment |
| Algebraic structure | square-ish | extreme aspect / low-rank | re-association of chains, factored forms, avoiding explicit products |
| Layout | contiguous | permuted/strided/channels-last | copy-then-compute vs stride-aware kernels |
| Dtype | fp32 | bf16/fp16/mixed | tensor-core paths, stable reformulations, accumulation precision |
| Hardware | HBM (A100/H100) | GDDR (L40S/4090) | roofline knee moves: same task, different answer |

### 3.2 Two-level generator (D-5)

**Level 1 — motif families.** ~10 hand-designed parameterised program families (e.g.
`ragged_attention_like`, `pairwise_topk`, `segment_reduce`, `embedding_bag_pool`,
`norm_act_chain`, `small_batched_linalg`, `moe_dispatch_combine`, `scan_recurrence`,
`matrix_chain`, `stencil/conv_like`). Each family declares its parameter space *and* ships 2–5
**reference strategies** — deliberately different approaches (e.g. "pad+SDPA", "packed Triton
varlen", "bucketed dispatch"). Strategies are not shown to agents; they exist to certify tasks.

**Level 2 — composition.** A typed DAG grammar composes motifs and glue ops (views, casts,
pointwise, gathers) into larger programs. Composition creates problems no single motif has:
fusion across motif boundaries, layout decisions that must be consistent across stages,
intermediates that should never exist. No hand-written optimum exists for these; they are
admitted by an automatic **headroom certificate** (below).

### 3.3 Certification: mechanical, on real hardware

A generated candidate becomes a benchmark task only if it passes all of:

1. **Oracle sanity** — reference is deterministic (or has bounded nondeterminism), fp64 re-run
   agrees within the declared tolerance, no NaN/Inf under the workload, memory fits with margin.
2. **Headroom certificate** — best automatic baseline is ≥ *H*× slower than an analytic lower
   bound: `max(bytes_touched / mem_bw, flops / peak_flops, n_min_kernels × launch_latency)`.
   If `torch.compile` is already near the roofline, there is nothing to engineer → reject.
   For Level-1 tasks the stronger form applies: some reference strategy beats the best automatic
   baseline by ≥ *S_min* (initially 1.3×) with CI excluding 1.0.
3. **Regime-flip certificate (family level)** — across the family's workload grid, the argmax
   reference strategy changes, and the *cross-regime penalty* (running regime A's best strategy
   on regime B) exceeds the timing noise floor by a wide margin (initially ≥ 1.25×). Task *pairs*
   straddling a boundary are emitted together.
4. **Measurement stability** — test–retest CV of the baseline timing below threshold on the
   target SKU; otherwise the task is too noisy to score.
5. **Anti-triviality** — a fixed battery of "lazy" solutions (wrap in `torch.compile`, add CUDA
   graphs, cast to bf16 where tolerance allows, `channels_last`) must not reach the certified
   best. What the lazy battery *does* achieve is folded into the baseline.

Difficulty is then a *measured* property (headroom, number of strategy classes needed, distance
from lazy battery to best-known), not an author's opinion.

## 4. The falsifiable core: transfer matrices

For a program *P* with workloads *A* and *B* on either side of a certified boundary, take any
solution optimized for *A* and evaluate it on *B*:

`regime_regret(A→B) = t(sol_A on B) / t(sol_B on B)`

- If regret ≈ 1 everywhere, workloads don't change what good looks like → **the central premise
  is false** and the project should pivot. This is measurable with reference strategies *before
  any agent runs*.
- If regret ≫ 1, then an agent that pattern-matches "attention → write FlashAttention" without
  measuring will be visibly punished on the sibling task, and *regime-appropriate strategy rate*
  becomes a clean behavioural metric of whether an agent reasons from profiler/runtime feedback.

This matrix is the scientific backbone of both the benchmark paper-claim and the quantization
study (feedback-use is otherwise very hard to measure).

## 5. Architecture (decoupled by on-disk contracts)

```
taskgen ──emits──▶ task package (dir, content-hashed, immutable)
                     task.toml  reference.py  workload.py  AGENT_README.md  [private/ strategies, certs]
runner  ──mounts public part into GPU sandbox; enforces budgets; records trajectory
agent   ──anything that can drive a shell; model = OpenAI-compatible URL
           └─ leaves ──▶ submission dir (solution.py + whatever it built)
evaluator(task package, submission dir, hardware) ──▶ result.json  (fresh sandbox, hidden seeds)
analysis  ──reads only result.json + trajectory.jsonl
```

- `taskgen` never imports `evaluator`; certification *calls* the evaluator CLI like anyone else.
- `evaluator` knows nothing about agents or how tasks were made.
- `agents` know nothing about scoring internals; they see `hotloop check` output only.
- The quantization study is a pure *client* of the benchmark: it changes the URL behind the agent.

## 6. MVP = smallest thing that can falsify the idea

1. Evaluator with trustworthy timing + anti-cheat battery, on one GPU SKU.
2. Three Level-1 families chosen to span different axes (proposed: `ragged_attention_like`
   [raggedness+materialization], `segment_reduce/embedding_bag` [index skew + reduction geometry],
   `norm_act_chain @ tiny-vs-huge` [launch vs bandwidth]) with 2–4 reference strategies each.
3. Certification run → transfer matrices. **Gate G1: do regime flips exist above noise?**
4. One minimal agent harness; one frontier model + one open-weight model on ~12 certified tasks.
   **Gate G2: does the benchmark discriminate, is it unsaturated, do agents show regime regret?**
5. Only then: Level-2 composition, second SKU, quantization ladder.

See [PLAN.md](PLAN.md) for gates and kill criteria.
