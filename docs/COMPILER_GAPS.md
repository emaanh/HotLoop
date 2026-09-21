# Compiler gaps: what HotLoop tasks are *for*

Status: v0 (2026-09-21). Direction set by Emaan: tasks should stretch kernel optimisation **beyond
what compilers can do, such that research on them can improve compilers**. Decision: D-33.

## The idea in one paragraph

A compiler makes the program you wrote fast; it does not change the program. Every large win in
dev-v0 (3.5–28× over `torch.compile`) came from *changing the program* in a way the compiler is
not allowed to, not informed enough to, or not built to. Where only code generation was at stake,
hand-written Triton beat the compiler by 1.05×. So "speedup over the compiler" is really a
measurement of a **gap in compiler capability**, and each task can say *which* gap. A benchmark
organised that way produces three things a compiler team can use: a map of where the gaps are and
how much they cost, a corpus of concrete before/after rewrites that closed them, and a regression
suite that shows a gap closing when a compiler release catches up.

## Gap taxonomy

Each family declares one primary gap class. "What would close it" is the compiler research the
task points at.

| # | Gap | Why a compiler doesn't do it today | What would close it | dev-v0 evidence |
|---|---|---|---|---|
| G1 | **Numerical contract** — transformations that change rounding | Re-association, reduction reordering, fused/mixed precision are illegal without permission; there is no way to tell a compiler "±1e-5 is fine" | *Tolerance-aware compilation*: a numerical contract as a first-class input. HotLoop already has one per task (fp64-calibrated tolerance) | matmul chains: compiler = eager at every shape; best order 4–28×; `multi_dot` exists but must be chosen by hand |
| G2 | **Data distribution** — structure only visible in the values | Compilers specialise on shapes and dtypes, never on "lengths are heavy-tailed" or "indices are Zipfian"; padding waste is invisible to them | *Distribution-guided compilation* (PGO for tensor programs): profile the data, specialise or multi-version the kernel | ragged pooling: 98% of the padded work is waste; 7–17×; the *right* kernel flips with the length distribution |
| G3 | **Algorithm substitution** | Replacing materialise-then-reduce by a streaming/online form, a loop by an associative scan, sort by select, needs algebraic facts (associativity, semiring structure) the IR doesn't carry | Algebraic-property inference + equality saturation with algorithm-level rules; "FlashAttention as a rewrite, not a library" | not yet in dev-v0 → planned families |
| G4 | **Cross-operator layout & materialisation** | Fusion stops at reductions, gathers and library calls; layout (packed vs padded, channels-last, AoS/SoA) is chosen per op, not per program | Whole-program layout planning; fusion across gather/reduce/GEMM boundaries | ragged pooling partially (gather → mask → softmax → weighted sum never needs the padded tensor) |
| G5 | **Host–device orchestration** | Launch-bound programs need horizontal fusion of many small tensors, sync removal, CUDA graphs under changing addresses | Automatic horizontal fusion; graph capture that tolerates fresh inputs without copying | controls at ~90 µs/call: compile's per-call guard cost is visible (0.64–0.73×); `reduce-overhead` loses to eager when every call has new input addresses |
| G6 | **Regime-dependent tuning** | Autotuners search a fixed config space at fixed shapes; the best block size / parallelisation axis moves with the workload | Autotuning over *workload* features, not just shapes; learned cost models | jagged kernel: BLOCK=16 vs 128 vs 1024 each win somewhere; wrong one costs up to 54× |
| G7 | **Global cost-model choices** | Recompute-vs-store, where to spend precision, when *not* to optimise | Program-level cost models with memory and accuracy budgets | controls: the right move is to stop; pattern-matched kernels score 0.03–0.8× |

## What this changes in the benchmark

1. **Every family carries a `gap_class`** (and optional secondary classes) in its provenance.
   Reports aggregate by gap: "G2 tasks: compiler leaves a median 9× on the table; agent recovers 95%".
2. **References must be natural.** A gap only matters to compiler people if real code looks like
   the reference. Padded ragged batches, left-to-right matmul chains, Python-loop recurrences and
   many-small-tensor updates all pass this test; adversarially slow references do not. Each family
   doc states where the pattern occurs in the wild.
3. **The lazy battery becomes "what the ecosystem already offers".** Besides `torch.compile`
   modes it must include the library answer when one exists — `multi_dot` (done), **nested/jagged
   tensors for ragged data (missing — to add)**, SDPA/FlexAttention for attention-like programs,
   `_foreach` ops for multi-tensor updates. A task only counts as a *gap* if the ecosystem's
   one-liner doesn't close it.
4. **A third score axis: closable-by-rule.** For each winning solution, record whether the rewrite
   is (a) a semantics-preserving rule a compiler could apply mechanically, (b) legal only under a
   numerical contract, (c) dependent on data-distribution knowledge, (d) a genuinely new algorithm.
   (a)–(c) are compiler to-do items; (d) is where agents add something compilers won't.
5. **The rewrite corpus.** For every solved task keep the (reference, workload, winning solution,
   speedup, gap class, closable-by) tuple. That is a dataset of optimisation *moves* with measured
   pay-offs — the raw material for rewrite rules, cost models and superoptimiser targets.
6. **Compiler progress tracker.** Re-run baselines on each PyTorch release. Headroom that shrinks
   is a gap the compiler closed; the benchmark regenerates harder instances of that class.
7. **Absolute scale.** Report fraction of the roofline bound next to speedup-over-compiler, so
   "gap" is measured against physics rather than against whatever reference strategy we wrote
   (the agent beat ours on every task; see LOG 2026-09-21).

## Families to build next, chosen to cover the empty rows

| Family | Gap | Natural reference | What a strong solution does | Regime flip to certify |
|---|---|---|---|---|
| `streaming_pairwise` | G3 | `softmax(f(Q,K)) @ V`, `logsumexp`/top-k over an N×M score matrix with a custom score SDPA can't express | tiled/online computation, never materialises N×M | small N·M → materialise wins; large → streaming; memory-capped variants |
| `linear_recurrence` | G3, G5 | `for t in range(T): h = a[t]*h + b[t]` | associative scan; chunked scan; CUDA graph for short T | short T & wide state → graph the loop; long T → scan |
| `multi_tensor_update` | G5 | per-parameter optimiser step over hundreds of small tensors | horizontal fusion / flat buffer | many tiny → flatten; few huge → per-tensor fused, flattening copy loses |
| `skewed_dispatch` | G2, G4 | MoE-style route → per-expert GEMM → combine, written with masks | sort + grouped GEMM vs padded batched GEMM | balanced vs skewed routing |
| `hot_row_gather` | G2 | embedding-bag with Zipfian indices | dedupe hot rows, cache, sort-based segment reduce | uniform vs Zipfian indices |
| `structured_algebra` | G1, G3 | `diag(A@B@C)`, Kronecker products, low-rank-plus-diagonal solves written densely | never forms the dense object | which factor is small decides the rewrite |
| Level-2 compositions | G4 | two motifs chained (ragged gather → chain → reduce) | one layout decision across stages | layout that is best per stage ≠ best overall |

## What we should be careful not to claim

- Beating the compiler here is not evidence the compiler is *bad*: it is doing a different job
  (D-32 shows compile == eager to within 1% once measured properly).
- A gap is relative to a compiler version and to the reference's naturalness. Both go in the record.
- dev-v0's strategy notes are public (D-28); gap-class *statistics* are safe to publish, per-task
  winning rewrites for held-out sets are not.
