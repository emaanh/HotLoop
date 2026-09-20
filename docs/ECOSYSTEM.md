# Ecosystem survey & reuse map

Status: **v1 (2026-09-20) — M0 verification done.** The landscape table came from a web survey;
the repos we intend to reuse were then **cloned and read** (two code-reading subagents, no code
run). Where the two disagree, the "Verified by reading the code" section below wins.
Still unverified: licences of TritonGym, TritonBench, AutoTriton, KernelGenBench, GSO, SWE-Perf.

## Landscape in one table

| Project | Formulation | Shapes | Baseline | Agent freedom | Anti-cheat | License | Take |
|---|---|---|---|---|---|---|---|
| [KernelBench](https://github.com/ScalingIntelligence/KernelBench) | 250 hand-written modules, single-shot | 1 fixed | eager | none | weak (many known exploits) | MIT | de facto standard; export compat only |
| [robust-kbench](https://github.com/SakanaAI/robust-kbench) (Sakana, post-incident) | ~13 tasks fwd+bwd | multi-config | eager + compile | — | output-variance checks, multi-init | Apache-2.0 | correctness-check ideas |
| [CUDA-L1](https://github.com/deepreinforce-ai/CUDA-L1), [Kevin-32B](https://arxiv.org/abs/2507.11948) | RL on KernelBench | fixed | eager(+compile) | multi-turn | exploit catalogues: side-stream timing (32.8% of kernels!), caching, try/except fallback, subclass reference | — | **exploit list → our anti-cheat tests** |
| [KernelGYM](https://github.com/hkust-nlp/KernelGYM) | RL env, 3 turns default, Triton | KernelBench's | torch ref | fixed API | Triton-launch instrumentation ("did your kernel run, what % of CUDA time"; substring matching, flawed denominator) | **README says Apache-2.0 but no LICENSE file** → re-implement only | worker isolation design; kernel-share check |
| [TritonBench](https://github.com/thunlp/TritonBench) | single-shot, scraped | — | — | none | harness bugs found by GEAK | ? | skip |
| [TritonGym](https://github.com/yil384/TritonGym-public) | agentic via fixed tools, 164 ops | multi | oracle Triton kernels | tool API, no shell | — | ? | idea: standardised tool budget |
| [Atrex-Bench](https://github.com/alibaba/atrex-bench) | 30 ops × 440 production-trace shapes | multi (traces) | torch.compile; **roofline fraction** | their agent | subprocess, freq lock, A-B-B-A | Apache-2.0 | **task dir format; ABBA timing; roofline.json** |
| [SOL-ExecBench](https://github.com/NVIDIA/SOL-ExecBench) / [SOLAR](https://github.com/NVlabs/SOLAR) | 235 problems from 124 models | dynamic multi | **roofline speed-of-light** | — | CUPTI all-stream kernel spans + kernel-count assertion, thread delta, id-pin monkeypatch check, data_ptr shifting. **No tolerance calibration, no explicit stream/cache detector** (verified) | Apache-2.0 | **anti-cheat + tolerance design; SOLAR for bounds if licence OK** |
| [AgentKernelArena](https://github.com/AMD-AGI/AgentKernelArena) | 426 task configs, real coding agents | held-out shapes (optional, manual, 4 families, not scored) | — | full agents (Claude Code, Codex…) | eval independent of agent | Apache-2.0 | agent-integration layer; held-out protocol. AMD-centric |
| [Apex](https://github.com/AMD-AGI/Apex) | agent pipeline, ROCm, e2e vLLM rescoring | — | — | heavily scaffolded (MCP servers) | AST scan | MIT | not neutral; skip |
| [GPU MODE kernelbot](https://github.com/gpu-mode/kernelbot) / reference-kernels | human+AI leaderboard, Modal H100 | multi | — | n/a | secret seeds, recheck after every timed run, adaptive repeats | **non-standard restrictive licence** | copy *design* of `eval.py`, not code |
| [FastKernels](https://github.com/Snowflake-AI-Research/fastkernels) | compositional L1–L4, traced shapes (**no nsys, no agent runtime** — verified) | multi | — | none (agent is external) | id-pin/thread/tensor-type checks | Apache-2.0 | timing primitive + hack checks |
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

## Verified by reading the code (M0, 2026-09-20)

HEADs read: atrex-bench 22d6a57, SOL-ExecBench a9fa080, SOLAR 62c5e34, robust-kbench 078f5ba,
reference-kernels f3295bb, KernelBench 423217d, mini-swe-agent 04d809c (v2.4.6), AgentKernelArena
0acf65b, fastkernels 9acebaa, KernelGYM 3a84417, CUDA-Agent 473025c.

### Corrections to the survey
| Claim | Reality |
|---|---|
| SOL-ExecBench calibrates tolerance by probing the reference | **Not in code.** Static `ToleranceSpec(1e-2, 1e-2, 99% matched)`, per-workload override. → our fp64 calibration is original, keep it. |
| SOL-ExecBench: stream monitoring, caching detection | No explicit detectors. Coverage is *implicit*: CUPTI spans all streams; asserts per-iteration kernel counts equal discovery counts. Inputs get a fresh `data_ptr` each iteration but **identical values** → a value-hash cache still hits. |
| SOL-ExecBench: 10 warmup + 3×50, locked clocks | 10 warmup + 50 iters once, median. Clock locking implemented but **off by default**. Public seed (200). |
| Atrex A-B-B-A interleaved timing | Real but **process-level**: 4 whole-pipeline worker runs (B,C,C,B), geomean. Not block-interleaved. Default timer is `do_bench(warmup=10, rep=100)` — those are *milliseconds* in Triton's API (unverified from memory), i.e. likely a latent bug. Seeds are derivable by the solver (sha256 of public strings). 1 correctness case by default. |
| Atrex clock locking | Real and good: `nvidia_clock.py` / `clock_lock.py` / `clock_monitor.py`, stdlib-only, SIGTERM-safe reset, in-run monitoring. |
| robust-kbench output range/std checks | They validate **tasks** (is the reference output near-constant? do inputs/inits matter?), not submissions. Useful for *our certification*. Also: runs candidate before *and* after the reference and requires both to match. |
| GPU MODE adaptive stop 0.1% vs 1% | Both: 0.1% SEM/mean in current evals, 1% in legacy ones. Cantor seed mixing and per-iteration fresh-data recheck confirmed. Licence = "Researcher Reciprocity v1.0", forbids training-related use downstream → design only. |
| FastKernels: nsys/Nsight integration, near-free-form agent | **Zero** nsys/nvtx references; **no agent runtime at all**. It's a bench harness (timing adapted from SOL-ExecBench). |
| AgentKernelArena ~196 tasks, isolated workspaces, held-out protocol | 426 configs. "Isolation" = copied dir + SHA-256 harness guard; agent container runs `--network=host`. Held-out shapes are an optional manual post-run step, LLM-generated, 4 families. Only budget is wall-clock. A *sidecar* eval container is properly hardened (`--network=none --cap-drop=ALL --read-only --security-opt=no-new-privileges`). |
| KernelGYM Apache-2.0 | README claims it; **no LICENSE file**. Launch hook monkeypatches Triton `JITFunction.launch/run/__getitem__`; CUDA-share uses substring name matching and adds CPU time to the denominator. No curated profiler summary for the model (TODO in code). |
| SOLAR usable on arbitrary fn/FX graph | No: needs KernelBench-style file, patched torchview, YAML between stages. Offline tool only. Atrex's `compute_roofline()` formula is the reusable part. |
| CUDA-Agent sandbox permission model | Prose in SKILL.md only; no enforcement in code. No licence. |

### What nobody does (confirmed gaps we fill)
- Secret seeds **and** fresh *values* every timed iteration **and** all-stream/device-wide timing, together.
- Tolerances derived from the reference's own numerical error.
- GPU-time budgets; network egress restriction for the agent; trusted evaluator on a copy the agent never touched (AgentKernelArena guards in-place with hashes instead).
- CIs / minimum effect sizes on speedups. Everyone reports a bare median or mean.

### Final reuse list (→ D-18, D-19)
| What | Source | How |
|---|---|---|
| Agent loop, Responses-API model class, retries, trajectory with raw responses | mini-swe-agent 2.4.6 (MIT) | **depend**, pinned. Write `SshDockerEnvironment` (~60 lines) + agent subclass for token/GPU budgets and per-command timing. Wrap every command in in-container `timeout -k` (upstream timeout orphans the in-container process). Head+tail truncation template from its swebench config. |
| Clock lock + monitor | atrex-bench (Apache-2.0) | **vendor** with attribution |
| Runtime hack guards: `elapsed_time` id pin, post-JIT thread-count delta, strict `type(t) is torch.Tensor`, evaluator-function id snapshot | SOL-ExecBench (Apache-2.0) | **vendor**/adapt |
| Shifting input pool (fresh `data_ptr`, no malloc in timed region) | SOL-ExecBench `io.py` (Apache-2.0) | **adapt** — and vary *values* too |
| CUPTI kernel-span timing + kernel-count assertion | SOL-ExecBench | **optional diagnostic** only: it excludes host time, which is part of our metric; pins cupti-python/CUDA 13 |
| Fresh-data recheck after each timed run; secret-seed mixing; adaptive stop | GPU MODE (design only) | **re-implement**; HMAC instead of Cantor; bootstrap-CI stop (already in `stats.py`) |
| Hardened evaluator container flags | AgentKernelArena sidecar | **re-implement** |
| Task-validity filters (near-constant output, inputs don't matter) | robust-kbench | **re-implement** inside certification |
| Candidate-before-and-after-reference check | robust-kbench | **re-implement** |
| Static regex pre-scan (try/except fallback, timing patches, streams, threads) | KernelBench (MIT) | **vendor** as a *flagging* signal, never as the verdict |
| Triton-launch hook + kernel time share | KernelGYM idea | **re-implement** (~100 lines, exact-name match, CUDA-only denominator) — later, as a "lazy optimisation" diagnostic |
| Roofline formula | atrex-bench `compute_roofline` | **vendor** when Level-2 needs it (parked) |
