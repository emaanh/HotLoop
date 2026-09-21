# Plan

Status: **executing M0.** Scope per D-17: M0 → M3-lite (OpenAI model only), then stop and report.
M4/M5 and open-weight models are parked. Updated 2026-09-20.
Cost detail and levers: [BUDGET.md](BUDGET.md).
Principle: every milestone ends in a **gate** with a number that can fail. Cheapest falsification first.

| M | Goal | Needs | Est. GPU $ | Status |
|---|---|---|---|---|
| M0 | Verify reuse targets by reading code; repo skeleton | nothing | 0 | ✅ done 2026-09-20 |
| M1 | Trustworthy measurement | Lambda | ~$50 (spent $0.82) | 🟢 G0 passed within- and cross-box (between-run SD ≈ 2%, one 5% outlier → D-22 pooling matters) |
| M2 | Regime flips exist (G1) | Lambda | ~$100 (spent ≈$2) | ✅ done 2026-09-21: G1 passes on evaluator-grade data; `benchmark/dev-v0` = 6 headroom + 2 diagnostic + 4 control tasks |
| M3 | Agents on MVP tasks (G2) — **M3-lite per D-17: OpenAI model only** | + OpenAI key | ~$350 + API | 🟡 runner, image, agent adapter built and verified; cost pilot running |
| M4 | Generator scale-up (Level 2, 2nd SKU) | — | ~$300 | ⏸ parked (D-17) |
| M5 | Quantization study | — | ~$1,400 (levers: BUDGET.md) | ⏸ parked (D-17) |
| M6 | Write-up | — | 0 | ⬜ |

## M0 — Ground truth on the ecosystem (no GPU)
- Clone & read: Atrex-Bench (task format, ABBA, roofline.json), SOL-ExecBench (anti-cheat,
  tolerance calibration, licence), AgentKernelArena (agent adapters), FastKernels, KernelGYM
  (launch instrumentation), robust-kbench, mini-swe-agent. Read GPU MODE `eval.py` for design only.
- Correct ECOSYSTEM.md where the survey was wrong. Fix licences.
- Repo skeleton: `uv` workspace with four packages (`taskgen`, `evaluator`, `runner`, `agents`) +
  `schemas/` (task.toml, result.json, trajectory.jsonl as JSON Schema). Schemas first.

## M1 — Measurement you can trust
- Evaluator v0: subprocess isolation, hidden seeds, ABBA interleave, adaptive repeats, bootstrap CI
  on ratio, environment fingerprint, fp64-calibrated tolerances.
- Anti-cheat regression suite: ≥8 null agents (EVAL.md) all score 0.
- Day-1 probes (ACCESS.md) on a Lambda A100 root VM (H100 when obtainable); SkyPilot provisioning with autostop.
- **Gate G0:** test–retest of the same solution across fresh sandboxes: speedup-ratio CV ≤ 3% for
  device-bound, ≤ 8% for launch-bound microbenchmarks. A deliberately planted 1.10× improvement is
  detected ≥ 95% of the time. *If it fails:* try another provider/SKU; then raise minimum effect sizes and drop
  launch-bound families.

## M2 — Do regime flips exist? (the core falsification, no agents involved)
- Three Level-1 families (DESIGN §6), each with 2–4 reference strategies and a workload grid of
  ~20–40 points. Sweep on A100-40GB (primary SKU, D-14) (+ H100 for a hardware-axis preview when capacity allows).
- Output: per-family strategy-ranking maps + transfer matrices + headroom over best auto baseline.
- **Gate G1 (kill criterion for D-1):** ≥ 2 of 3 families show (a) argmax-strategy change across
  the grid, (b) cross-regime penalty ≥ 1.25× with CI excluding 1.1, (c) ≥ 1.3× headroom over the
  strongest automatic baseline in ≥ 2 regimes. *If it fails:* torch.compile is better than
  assumed → pivot candidates: hardware-flip tasks, memory-capped tasks, Level-2 composition where
  compiler fusion heuristics break, or backward passes.
- Emit ~12 certified tasks (6 sibling pairs).

## M3 — Does the benchmark measure agents? 
- Runner v0: Docker-over-SSH on SkyPilot-provisioned VMs (D-13); minimal bash-loop harness; trajectory logging with raw completions.
- Pilot: frontier model (OpenAI, D-15) + Qwen3.8-27B BF16 (via OpenRouter or a vLLM serving VM) + a trivial
  "lazy" scripted agent, 12 tasks × 3 seeds.
- **Gate G2:** (a) unsaturated: frontier headroom_closed < 0.9 on most tasks; (b) not at floor:
  open model gets > 0 on ≥ 1/3 of tasks; (c) discriminative: frontier > open > lazy with
  non-overlapping CIs; (d) regime regret observed in ≥ some agent solutions (i.e. sibling tasks get
  the same strategy and pay for it); (e) zero successful reward hacks on manual audit of top
  scores (every > 2× result read by a human/Claude).
- Pilot variance → power analysis → final N for M5. Tune budgets (D-6).

## M4 — Scale the generator
- Families → ~8–10. Level-2 composition grammar + automatic headroom certificate (roofline bound
  via SOLAR or own FX-graph counter). Second SKU (H100; GDDR/Blackwell part via another provider if worthwhile): same task, different answer.
- Difficulty calibration from measured properties. Freeze `hotloop-v0` task set (content-hashed),
  ~40–60 tasks, with a held-out generator seed for contamination-free regeneration.

## M5 — Quantization study
- Pre-register H1–H5 + analysis in DECISIONS. Phase-1 arms (QUANT_STUDY.md): BF16×2, FP8,
  GPTQ-4, AWQ-4 × ~24 tasks × N seeds. Serving on a dedicated H100 VM, benchmark VMs separate.
- Analyse: score, headroom, per-channel failures, recovery, regime regret, horizon interaction,
  tokens/turn. Decide Phase 2 (NVFP4, W8A8, KV-FP8, sub-4-bit via llama.cpp bridge) from results.

## M6 — Write-up
Paper-style report in `docs/paper/`: generator + certification, measurement validity study,
transfer matrices, agent results, quant study, threats to validity, all negative results.

## Standing risks
| Risk | Mitigation |
|---|---|
| torch.compile max-autotune already near-optimal on motifs | G1 tests this first; pivots listed |
| Cloud timing noise | G0; root VMs with locked clocks; ratios not absolutes |
| Leaked VMs / Lambda capacity | SkyPilot autostop + watchdog + spend log; provider-agnostic runner |
| Reward hacking | trusted separate evaluator; exploit regression suite; audit all > 2× |
| Open model at floor | G2(b); easier tier via larger headroom tasks; bigger model |
| Reference strategies encode *my* biases about what's optimal | best-known is a moving max over all submissions; certificates need only *a* flip, not the global optimum |
| Autotune/compile time eats agent budget | budget GPU-seconds separately for build vs run; pilot in M3 |
| Hybrid-GDN vLLM bugs | fallback Qwen3.6-27B; pin engine build |
