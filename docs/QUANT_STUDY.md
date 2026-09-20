# Study: does quantization degrade autonomous performance-engineering ability?

Status: v0 draft (2026-09-20). Model/precision ladder pending research → see "Model selection".
Runs only after benchmark gates G1–G2 pass ([PLAN.md](PLAN.md)).

## Question

Static benchmarks say 8-bit is ~lossless and 4-bit costs a point or two. Long-horizon agentic
work compounds small per-token errors over 50–150 turns and 30–100k tokens of context, and
depends on abilities (using feedback, abandoning a bad idea) that short benchmarks do not
exercise. Does precision reduction change *how* an agent engineers, not just its final score?

## Hypotheses (to be pre-registered in DECISIONS before the runs)

- **H1 (score):** final task score degrades monotonically with precision; W8/FP8 within noise of
  BF16; ≤4-bit measurably worse.
- **H2 (discovery):** lower precision reduces the *diversity* of strategy classes attempted and
  the rate of finding the regime-appropriate one, more than it reduces the ability to implement a
  given strategy correctly.
- **H3 (recovery):** conditional on a failed step (compile error, wrong output, slowdown), lower
  precision raises perseveration (re-trying near-identical actions) and lowers P(recover).
- **H4 (feedback use):** on certified regime-flip *pairs*, lower precision lowers the rate at
  which the agent's strategy differs appropriately between siblings (higher regime regret) —
  i.e. it pattern-matches the program rather than reading the profile.
- **H5 (horizon):** degradation grows with turn index / context length (interaction term), and
  is worsened by KV-cache quantization independently of weight quantization.

Each is falsifiable; "no detectable difference at our power" is a reportable result, with the
minimum detectable effect stated.

## Design

- **Controlled variable:** weight (and separately KV-cache) precision of *one* model family.
  Same harness, prompts, sampling params, serving engine + version, tool schema, budgets.
- **Unit:** trajectory = (task, precision, seed). Paired across precisions by task and seed.
- **Size (initial):** ~24 certified tasks (12 regime-flip pairs) × 4–5 precisions × 3 seeds
  ≈ 290–360 trajectories per model. Power analysis after G2 pilot gives real variance.
- **Analysis:** mixed-effects models (task random intercept), paired bootstrap on score deltas,
  survival-style analysis for time-to-first-correct-speedup; all from `result.json` +
  `trajectory.jsonl`. No LLM-judged primary metrics; LLM labelling (strategy class of each
  attempt) only as secondary, validated against a hand-labelled sample.

## Metrics from trajectories (mechanical where possible)

| Construct | Measure |
|---|---|
| Outcome | score, headroom_closed, P(correct final), P(any speedup > 1) |
| Discovery | # distinct solution strategies that reached `hotloop check`; best-so-far curve vs turn/GPU-sec |
| Recovery | after failing check/compile: P(next success within k turns), edit-distance between consecutive failed attempts (perseveration), revert-to-known-good rate |
| Feedback use | fraction of turns invoking timer/profiler; whether next edit touches the op the profile ranked top; regime regret on sibling tasks |
| Hygiene | malformed tool calls, syntax-invalid code, hallucinated APIs (ImportError/AttributeError rate), final-answer-worse-than-best-seen rate |
| Horizon | all of the above as a function of turn index and context tokens |

## Confounds to control

- Quantization *method* vs *bit-width* (AWQ vs GPTQ vs NVFP4 at 4-bit): include ≥2 methods at
  one bit-width to separate them.
- Natively low-precision models (e.g. MXFP4-trained) have no honest BF16 baseline → not the
  primary family.
- Serving kernel differences changing sampling nondeterminism: fix seeds, log engine build,
  run BF16 twice to measure the *null* between-run variance.
- Throughput differences: budgets are in **turns/tokens and GPU-seconds in the sandbox**, not
  wall-clock, so faster quantized serving doesn't buy more attempts. (A secondary
  "equal-dollar" analysis can ask the practical question: is 4-bit with 2× the attempts better?)
- Tool-call parser brittleness unrelated to capability: use a minimal text/bash protocol that
  all precisions parse identically; count format failures separately.

## Model selection (D-9)

Criteria: (1) strong enough to score non-trivially at BF16 (else floor effects hide everything),
(2) honest BF16 master weights + reputable quantized checkpoints, (3) BF16 fits 1× 80GB so the
whole ladder runs on identical hardware at TP=1, (4) reliable tool use in vLLM.

**Primary: `Qwen/Qwen3.8-27B`** (dense, Apache-2.0, BF16-native, ~55 GB; hybrid GDN/full-attn so
KV is small ≈ 6.5 GB per 100k tokens). Fallback **`Qwen/Qwen3.6-27B`** (same arch, better-tested
serving) if vLLM issue #55766 (NaN logits after prefix-cache hits on 3.8 GDN) bites — prefix
caching is ~5× on serving cost for long trajectories, so we can't just turn it off.
**Replication (later): `Qwen/Qwen3.6-35B-A3B`** for a MoE contrast.
**Excluded from the ladder:** gpt-oss (MXFP4-native), Devstral-Small-2 (FP8-native), Kimi
(INT4-native), Qwen3-Coder-Next (BF16 needs 4 GPUs). A natively-low-precision model may be added
as a single reference point, never as a ladder.
A frontier API model runs once as an unquantized ceiling to show the benchmark is unsaturated.

_Source: subagent survey of HF + vLLM docs, 2026-09-20; checkpoint names to be re-verified when
we pull them. Vendor benchmark numbers not independently checked._

### Arms

Phase 1 (vLLM, H100, TP=1, KV cache BF16, spec-decoding off, same parser/template, pinned build):

| Arm | Checkpoint | Purpose |
|---|---|---|
| A0 | BF16 original — **run twice** | baseline + null distribution (between-run variance) |
| A1 | official `-FP8` (block-128) | "free lunch" claim |
| A2 | RedHatAI W4A16 GPTQ | mainstream 4-bit |
| A3 | cyankiwi W4A16 AWQ | method-vs-bitwidth control at 4 bits |

Phase 2 (only if Phase 1 shows an effect or a clean null worth extending):
- A4 NVFP4 (weight-only on H100; note different arithmetic on Blackwell)
- A5 INT8 W8A8 (self-quantized with llm-compressor; also lets us test code-calibrated vs
  chat-calibrated quantization — public 4-bit checkpoints are calibrated on 4k-token chat data)
- KV-cache factor: FP8 KV × {A0, A2}
- Sub-4-bit (Q3_K_XL, Q2_K_XL) needs llama.cpp → engine confound → requires Q8_0 and Q4_K_M
  *bridge arms* in llama.cpp to separate engine from bit-width. Expensive; do last.

### Prior work that shapes the metrics

- Reasoning: W8/W4A16 ≈ lossless on static tasks; damage grows with difficulty (2504.04823).
  Errors are execution-type, early, and cascade (2505.11574). Quantized reasoners **abandon
  correct mid-chain answers** and think longer (2606.00206, 2606.25519) → we must report tokens
  per turn and watch for abandoning known-good solutions (our "final worse than best-seen" rate).
- Long context: 4-bit can lose heavily > 64k (2505.20276), contested by vendor RULER numbers →
  our horizon interaction (H5) is a live question.
- Agents: "Flat Score, Amplified Failures" (2607.27275) — headline score flat across 16/8/4-bit
  on tau2-bench while existing failure modes amplify (tool-name hallucination ↑ 2.5×) →
  **per-channel failure reporting is mandatory; a flat headline score is not a null result.**
- Unstudied (our opening): multi-turn recovery and feedback use; 50–150-turn growing-context
  trajectories; graded (speedup) objectives rather than pass/fail; any GPU-kernel benchmark
  across precisions.

### Serving pitfalls to engineer around

- Log **raw completions**; score malformed tool calls ourselves, independent of vLLM's parser
  (vLLM #39056 drops tool calls emitted inside `<think>` in non-streaming mode).
- Force `--kv-cache-dtype auto` (RedHat INT4 ships FP8 KV scales that could silently activate).
- Qwen discourages greedy decoding in thinking mode → temperature 1.0-ish, ≥3–5 seeds, trajectory
  as unit. Fix `reasoning_effort`. Per-request seeds; evaluate `VLLM_BATCH_INVARIANT=1`.
- Serving GPU ≠ benchmark GPU, always.
- Est. serving cost ≈ 3–5 H100-min per trajectory with prefix caching → ~$30–100 per arm of 300
  trajectories. Sandbox GPU time will dominate total cost.
