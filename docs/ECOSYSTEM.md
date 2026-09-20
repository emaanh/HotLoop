# Ecosystem survey & reuse map

Status: v0 (2026-09-20). Source: web survey by a research subagent (READMEs + paper abstracts via
a summarising fetcher). **Treat details as leads, not facts** until verified by reading the code —
verification of the repos we intend to reuse is milestone M0 in [PLAN.md](PLAN.md).
Unverified: licenses of TritonGym, TritonBench, AutoTriton, SOL-ExecBench, KernelGenBench, GSO,
SWE-Perf; commit recency for all.

## Landscape in one table

| Project | Formulation | Shapes | Baseline | Agent freedom | Anti-cheat | License | Take |
|---|---|---|---|---|---|---|---|
| [KernelBench](https://github.com/ScalingIntelligence/KernelBench) | 250 hand-written modules, single-shot | 1 fixed | eager | none | weak (many known exploits) | MIT | de facto standard; export compat only |
| [robust-kbench](https://github.com/SakanaAI/robust-kbench) (Sakana, post-incident) | ~13 tasks fwd+bwd | multi-config | eager + compile | — | output-variance checks, multi-init | Apache-2.0 | correctness-check ideas |
| [CUDA-L1](https://github.com/deepreinforce-ai/CUDA-L1), [Kevin-32B](https://arxiv.org/abs/2507.11948) | RL on KernelBench | fixed | eager(+compile) | multi-turn | exploit catalogues: side-stream timing (32.8% of kernels!), caching, try/except fallback, subclass reference | — | **exploit list → our anti-cheat tests** |
| [KernelGYM](https://github.com/hkust-nlp/KernelGYM) | RL env, ≤3 turns, Triton | KernelBench's | torch ref | fixed API | Triton-launch instrumentation ("did your kernel run, what % of CUDA time") | Apache-2.0 | worker isolation design; kernel-share check |
| [TritonBench](https://github.com/thunlp/TritonBench) | single-shot, scraped | — | — | none | harness bugs found by GEAK | ? | skip |
| [TritonGym](https://github.com/yil384/TritonGym-public) | agentic via fixed tools, 164 ops | multi | oracle Triton kernels | tool API, no shell | — | ? | idea: standardised tool budget |
| [Atrex-Bench](https://github.com/alibaba/atrex-bench) | 30 ops × 440 production-trace shapes | multi (traces) | torch.compile; **roofline fraction** | their agent | subprocess, freq lock, A-B-B-A | Apache-2.0 | **task dir format; ABBA timing; roofline.json** |
| [SOL-ExecBench](https://github.com/NVIDIA/SOL-ExecBench) / [SOLAR](https://github.com/NVlabs/SOLAR) | 235 problems from 124 models | dynamic multi | **roofline speed-of-light** | — | strongest found: stream/thread monitor, cache + monkeypatch detection, addr randomisation; tolerance calibrated by probing reference | ? | **anti-cheat + tolerance design; SOLAR for bounds if licence OK** |
| [AgentKernelArena](https://github.com/AMD-AGI/AgentKernelArena) | 196 tasks, real coding agents | **held-out shapes** | — | full agents (Claude Code, Codex…) | eval independent of agent | Apache-2.0 | agent-integration layer; held-out protocol. AMD-centric |
| [Apex](https://github.com/AMD-AGI/Apex) | agent pipeline, ROCm, e2e vLLM rescoring | — | — | heavily scaffolded (MCP servers) | AST scan | MIT | not neutral; skip |
| [GPU MODE kernelbot](https://github.com/gpu-mode/kernelbot) / reference-kernels | human+AI leaderboard, Modal H100 | multi | — | n/a | secret seeds, recheck after every timed run, adaptive repeats | **non-standard restrictive licence** | copy *design* of `eval.py`, not code |
| [FastKernels](https://github.com/Snowflake-AI-Research/fastkernels) | compositional L1–L4, traced shapes, nsys | multi | — | closest to free-form | — | Apache-2.0 | read closely in M0 |
| [BackendBench](https://github.com/meta-pytorch/BackendBench) | op correctness via OpInfo | — | — | — | — | BSD-ish | edge-case input source |
| DRTriton ([arXiv 2603.21465](https://arxiv.org/abs/2603.21465)) | **procedural operator-DAG sampler (CSP)** — for *training data* | — | — | — | — | no code found | closest prior art for Level-2 grammar; cite |
| GSO / SWE-Perf / SWE-fficiency | CPU repo-level perf, free-form shell | hidden workloads | expert patch | full | — | — | scoring vs best-known; **reliability study: only 39/102 GSO tasks survive cross-machine replay** |

Other 2026 signals: KernelBenchX (46.6% of *correct* kernels slower than eager; speedups vary 21×
across hardware; iteration improves correctness not speed), KernelGenBench ("ghost replay",
profiler-signature checks).

## Gap (what HotLoop is for)

1. **No benchmark generates its tasks.** All are fixed sets → contamination/saturation. DRTriton
   samples programs, but for training and without performance structure.
2. **Shape variation is used for generalisation testing, never as the object of study.** Nobody
   constructs tasks so that the regime *flips the optimal strategy*, so nobody can score whether
   the agent *diagnosed* the regime. ← our core contribution (DESIGN §3–4).
3. **Free-form shell + real profilers + budget + separate trusted evaluator** does not exist as a
   neutral benchmark (FastKernels/AgentKernelArena are nearest).
4. **Statistics are thin**: almost no CIs, minimum effect sizes, or cross-machine validation.

Things we should *not* claim as novel: multi-shape eval, roofline scoring, torch.compile baseline,
anti-cheat, production-trace shapes. Those are solved; we adopt them.

## Reuse decisions (→ D-7)

| Need | Decision |
|---|---|
| Task package layout | Atrex-style dir (`reference.py`, `input.py`→our `workload.py`, hidden metadata/roofline). Add KernelBench-compatible export later for comparability. |
| Timing | Own thin implementation following GPU MODE `eval.py` design (adaptive repeats, secret seeds, recheck after timed runs) + Atrex ABBA + clock locking. `triton.testing.do_bench` as primitive where adequate. |
| Anti-cheat | Union of known exploit catalogues (CUDA-L1, Kevin, Sakana, SOL-ExecBench, KernelGYM kernel-share) as a *regression test suite*; see EVAL.md. |
| Tolerances | SOL-ExecBench-style calibration by probing the reference (matches our fp64 idea). |
| Edge inputs | BackendBench/OpInfo where ops overlap. |
| Roofline lower bound | Try SOLAR (licence check); else simple analytic counter over the FX graph. |
| Execution backend | **Don't** adopt KernelGYM's FastAPI+Redis for MVP — one sandbox per trajectory from a cloud sandbox API is simpler. Revisit if we do RL-scale throughput. |
| Agent harness | mini-swe-agent-style bash loop for the controlled study; AgentKernelArena-style adapters for Claude Code/Codex as uncontrolled references. |
| Build ourselves | regime-flip generator + certification, transfer-matrix scoring, free-form sandbox contract, trajectory analytics. |
